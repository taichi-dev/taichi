from taichi._lib import core as _ti_core
from taichi.aot.utils import produce_injected_args
from taichi.lang import kernel_impl
from taichi.lang._ndarray import Ndarray
from taichi.lang.exception import TaichiRuntimeError

ArgKind = _ti_core.ArgKind
Arg = _ti_core.Arg


def gen_cpp_kernel(kernel_fn, args):
    kernel = kernel_fn._primal
    assert isinstance(kernel, kernel_impl.Kernel)
    injected_args = produce_injected_args(kernel, symbolic_args=args)
    kernel.ensure_compiled(*injected_args)
    return kernel.kernel_cpp


class Sequential:
    def __init__(self, seq):
        self.seq_ = seq

    def dispatch(self, kernel_fn, *args):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args)
        self.seq_.dispatch(kernel_cpp, args)


class Graph:
    def __init__(self):
        self._graph_builder = _ti_core.GraphBuilder()
        self._compiled_graph = None

    def dispatch(self, kernel_fn, *args):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args)
        self._graph_builder.dispatch(kernel_cpp, args)

    def create_sequential(self):
        return Sequential(self._graph_builder.create_sequential())

    def append(self, node):
        # TODO: support appending dispatch node as well.
        assert isinstance(node, Sequential)
        self._graph_builder.seq().append(node.seq_)

    def compile(self):
        self._compiled_graph = self._graph_builder.compile()

    def run(self, args):
        arg_ptrs = {}
        # Only support native python numerical types (int, float) for now.
        arg_ints = {}
        arg_floats = {}

        for k, v in args.items():
            if isinstance(v, Ndarray):
                arg_ptrs[k] = v.arr
            elif isinstance(v, int):
                arg_ints[k] = v
            elif isinstance(v, float):
                arg_floats[k] = v
            else:
                raise TaichiRuntimeError(
                    'Only python int, float and ti.Ndarray are supported as runtime arguments'
                )
        self._compiled_graph.run(arg_ptrs, arg_ints, arg_floats)


__all__ = ['Graph', 'Arg', 'ArgKind']
