from taichi._lib import core as _ti_core
from taichi.aot.utils import produce_injected_args
from taichi.lang import kernel_impl
from taichi.lang._ndarray import Ndarray
from taichi.lang.exception import TaichiRuntimeError
from taichi.lang.matrix import Matrix, MatrixType

ArgKind = _ti_core.ArgKind


def gen_cpp_kernel(kernel_fn, args):
    kernel = kernel_fn._primal
    assert isinstance(kernel, kernel_impl.Kernel)
    injected_args = produce_injected_args(kernel, symbolic_args=args)
    key = kernel.ensure_compiled(*injected_args)
    return kernel.compiled_kernels[key]


class Sequential:
    def __init__(self, seq):
        self.seq_ = seq

    def dispatch(self, kernel_fn, *args):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args)
        self.seq_.dispatch(kernel_cpp, args)


class GraphBuilder:
    def __init__(self):
        self._graph_builder = _ti_core.GraphBuilder()

    def dispatch(self, kernel_fn, *args):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args)
        unzipped_args = []
        # Tuple for matrix args
        # FIXME remove this when native Matrix type is ready
        for arg in args:
            if isinstance(arg, list):
                for sublist in arg:
                    unzipped_args.extend(sublist)
            else:
                unzipped_args.append(arg)
        self._graph_builder.dispatch(kernel_cpp, unzipped_args)

    def create_sequential(self):
        return Sequential(self._graph_builder.create_sequential())

    def append(self, node):
        # TODO: support appending dispatch node as well.
        assert isinstance(node, Sequential)
        self._graph_builder.seq().append(node.seq_)

    def compile(self):
        return Graph(self._graph_builder.compile())


class Graph:
    def __init__(self, compiled_graph) -> None:
        self._compiled_graph = compiled_graph

    def run(self, args):
        # Support native python numerical types (int, float), Ndarray.
        # Taichi Matrix types are flattened into (int, float) arrays.
        # TODO diminish the flatten behavior when Matrix becomes a Taichi native type.
        arg_ptrs = {}
        arg_ints = {}
        arg_floats = {}
        arg_doubles = {}

        for k, v in args.items():
            if isinstance(v, Ndarray):
                arg_ptrs[k] = v.arr
            elif isinstance(v, int):
                arg_ints[k] = v
            elif isinstance(v, float):
                arg_floats[k] = v
            elif isinstance(v, Matrix):
                mat_val_id = 0
                for a in range(v.n):
                    for b in range(v.m):
                        key = f"{k}_mat_arg_{mat_val_id}"
                        mat_val_id += 1
                        if isinstance(v[a, b], int):
                            arg_ints[key] = int(v[a, b])
                        elif isinstance(v[a, b], float):
                            arg_floats[key] = float(v[a, b])
                        else:
                            raise TaichiRuntimeError(
                                f'Only python int, float are supported as matrix runtime arguments but got {type(v)}'
                            )
            else:
                raise TaichiRuntimeError(
                    f'Only python int, float and ti.Ndarray are supported as runtime arguments but got {type(v)}'
                )
        self._compiled_graph.run(arg_ptrs, arg_ints, arg_floats, arg_doubles)


def Arg(tag, name, dtype, field_dim=0, element_shape=()):
    if isinstance(dtype, MatrixType):
        if len(element_shape) > 0:
            raise TaichiRuntimeError(
                f'Element shape for MatrixType argument "{name}" is not supported.'
            )
        mat_type = dtype
        arg_list = []
        i = 0
        for _ in range(mat_type.n):
            arg_sublist = []
            for _ in range(mat_type.m):
                arg_sublist.append(
                    _ti_core.Arg(tag, f'{name}_mat_arg_{i}', dtype.dtype,
                                 field_dim, element_shape))
                i += 1
            arg_list.append(arg_sublist)
        return arg_list

    return _ti_core.Arg(tag, name, dtype, field_dim, element_shape)


__all__ = ['GraphBuilder', 'Graph', 'Arg', 'ArgKind']
