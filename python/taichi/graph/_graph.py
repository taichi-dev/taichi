import warnings

from taichi._lib import core as _ti_core
from taichi.aot.utils import produce_injected_args
from taichi.lang import kernel_impl
from taichi.lang._ndarray import Ndarray
from taichi.lang._texture import Texture
from taichi.lang.exception import TaichiRuntimeError
from taichi.lang.matrix import Matrix, MatrixType

ArgKind = _ti_core.ArgKind


def gen_cpp_kernel(kernel_fn, args):
    kernel = kernel_fn._primal
    assert isinstance(kernel, kernel_impl.Kernel)
    injected_args = produce_injected_args(kernel, symbolic_args=args)
    key = kernel.ensure_compiled(*injected_args)
    return kernel.compiled_kernels[key]


def flatten_args(args):
    unzipped_args = []
    # Tuple for matrix args
    # FIXME remove this when native Matrix type is ready
    for arg in args:
        if isinstance(arg, list):
            for sublist in arg:
                unzipped_args.extend(sublist)
        else:
            unzipped_args.append(arg)
    return unzipped_args


class Sequential:
    def __init__(self, seq):
        self.seq_ = seq

    def dispatch(self, kernel_fn, *args):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args)
        unzipped_args = flatten_args(args)
        self.seq_.dispatch(kernel_cpp, unzipped_args)


class GraphBuilder:
    def __init__(self):
        self._graph_builder = _ti_core.GraphBuilder()

    def dispatch(self, kernel_fn, *args):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args)
        unzipped_args = flatten_args(args)
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
        flattened = {}
        for k, v in args.items():
            if isinstance(v, Ndarray):
                flattened[k] = v.arr
            elif isinstance(v, Texture):
                flattened[k] = v.tex
            elif isinstance(v, Matrix):
                mat_val_id = 0
                for a in range(v.n):
                    for b in range(v.m):
                        key = f"{k}_mat_arg_{mat_val_id}"
                        mat_val_id += 1
                        if getattr(v, "ndim", 2) == 2:
                            flattened[key] = v[a, b]
                        else:
                            flattened[key] = v[a]
            elif isinstance(v, (int, float)):
                flattened[k] = v
            else:
                raise TaichiRuntimeError(
                    f'Only python int, float, ti.Matrix and ti.Ndarray are supported as runtime arguments but got {type(v)}'
                )
        self._compiled_graph.run(flattened)


def Arg(tag,
        name,
        dtype=None,
        ndim=0,
        field_dim=None,
        element_shape=(),
        channel_format=None,
        shape=(),
        num_channels=None):
    if field_dim is not None:
        if ndim != 0:
            raise TaichiRuntimeError(
                'field_dim is deprecated, please do not specify field_dim and ndim at the same time.'
            )
        warnings.warn(
            "The field_dim argument for ndarray will be deprecated in v1.5.0, use ndim instead.",
            DeprecationWarning)
        ndim = field_dim

    if tag == ArgKind.SCALAR:
        # The scalar tag should never work with array-like parameters
        if ndim > 0 or isinstance(dtype, MatrixType) or len(element_shape) > 0:
            raise TaichiRuntimeError(
                f'Illegal Arg parameter (dtype={dtype}, ndim={ndim}, element_shape={element_shape}) for Scalar tag.'
            )
        return _ti_core.Arg(tag, name, dtype, ndim, element_shape)

    if tag == ArgKind.NDARRAY:
        # Ndarray with matrix data type
        if isinstance(dtype, MatrixType):
            return _ti_core.Arg(tag, name, dtype.dtype, ndim,
                                dtype.get_shape())
        # Ndarray with scalar data type
        if len(element_shape) > 0:
            warnings.warn(
                "The element_shape argument for ndarray will be deprecated in v1.5.0, use vector or matrix data type instead.",
                DeprecationWarning)
        return _ti_core.Arg(tag, name, dtype, ndim, element_shape)

    if tag == ArgKind.MATRIX:
        if not isinstance(dtype, MatrixType):
            raise TaichiRuntimeError(
                f'Tag {tag} must specify matrix data type, but got {dtype}.')
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
                    _ti_core.Arg(tag, f'{name}_mat_arg_{i}', dtype.dtype, ndim,
                                 element_shape))
                i += 1
            arg_list.append(arg_sublist)
        return arg_list

    if tag == ArgKind.TEXTURE or tag == ArgKind.RWTEXTURE:
        if channel_format is None or len(shape) == 0 or num_channels is None:
            raise TaichiRuntimeError(
                'channel_format, num_channels and shape arguments are required for texture arguments'
            )
        return _ti_core.Arg(tag,
                            name,
                            channel_format=channel_format,
                            num_channels=num_channels,
                            shape=shape)
    raise TaichiRuntimeError(f'Unknowm tag {tag} for graph Arg {name}.')


__all__ = ['GraphBuilder', 'Graph', 'Arg', 'ArgKind']
