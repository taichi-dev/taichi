from taichi._lib import core as _ti_core
from taichi.aot.utils import produce_injected_args
from taichi.lang import kernel_impl

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
        arg_ptrs = {k: args[k].arr for k in args.keys()}
        self._compiled_graph.run(arg_ptrs)


__all__ = ['Graph', 'Arg', 'ArgKind']
