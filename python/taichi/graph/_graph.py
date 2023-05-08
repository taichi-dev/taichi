import warnings
from typing import Any, Dict, List

from taichi._lib import core as _ti_core
from taichi.aot.utils import produce_injected_args
from taichi.lang import enums, impl, kernel_impl
from taichi.lang._ndarray import Ndarray
from taichi.lang._texture import Texture
from taichi.lang.exception import TaichiRuntimeError
from taichi.lang.matrix import Matrix, MatrixType
from taichi.types.texture_type import FORMAT2TY_CH, TY_CH2FORMAT

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
                    f"Only python int, float, ti.Matrix and ti.Ndarray are supported as runtime arguments but got {type(v)}"
                )
        self._compiled_graph.jit_run(impl.get_runtime().prog.config(), flattened)


def _deprecate_arg_args(kwargs: Dict[str, Any]):
    if "field_dim" in kwargs:
        warnings.warn(
            "The field_dim argument for ndarray will be deprecated in v1.6.0, use ndim instead.",
            DeprecationWarning,
        )
        if "ndim" in kwargs:
            raise TaichiRuntimeError(
                "field_dim is deprecated, please do not specify field_dim and ndim at the same time."
            )
        kwargs["ndim"] = kwargs["field_dim"]
        del kwargs["field_dim"]
    tag = kwargs["tag"]

    if tag == ArgKind.SCALAR:
        if "element_shape" in kwargs:
            warnings.warn(
                "The element_shape argument for scalar will be deprecated in v1.6.0. You can remove them safely.",
                DeprecationWarning,
            )
            del kwargs["element_shape"]

    if tag == ArgKind.NDARRAY:
        if "element_shape" not in kwargs:
            if "dtype" in kwargs:
                dtype = kwargs["dtype"]
                if isinstance(dtype, MatrixType):
                    kwargs["dtype"] = dtype.dtype
                    kwargs["element_shape"] = dtype.get_shape()
                else:
                    kwargs["element_shape"] = ()
        else:
            warnings.warn(
                "The element_shape argument for ndarray will be deprecated in v1.6.0, use vector or matrix data type instead.",
                DeprecationWarning,
            )
            if "dtype" not in kwargs:
                dtype = kwargs["dtype"]
                if isinstance(dtype, MatrixType):
                    raise TaichiRuntimeError("Please do not specify element_shape when dtype is a matrix type.")

    if tag == ArgKind.RWTEXTURE or tag == ArgKind.TEXTURE:
        if "dtype" in kwargs:
            warnings.warn(
                "The dtype argument for texture will be deprecated in v1.6.0, use format instead.",
                DeprecationWarning,
            )
            del kwargs["dtype"]

        if "shape" in kwargs:
            warnings.warn(
                "The shape argument for texture will be deprecated in v1.6.0, use ndim instead. (Note that you no longer need the exact texture size.)",
                DeprecationWarning,
            )
            kwargs["ndim"] = len(kwargs["shape"])
            del kwargs["shape"]

        if "channel_format" in kwargs or "num_channels" in kwargs:
            if "fmt" in kwargs:
                raise TaichiRuntimeError(
                    "channel_format and num_channels are deprecated, please do not specify channel_format/num_channels and fmt at the same time."
                )
            if tag == ArgKind.RWTEXTURE:
                fmt = TY_CH2FORMAT[(kwargs["channel_format"], kwargs["num_channels"])]
                kwargs["fmt"] = fmt
                warnings.warn(
                    "The channel_format and num_channels arguments for texture will be deprecated in v1.6.0, use fmt instead.",
                    DeprecationWarning,
                )
            else:
                warnings.warn(
                    "The channel_format and num_channels arguments are no longer required for non-RW textures since v1.6.0, you can remove them safely.",
                    DeprecationWarning,
                )
            if "channel_format" in kwargs:
                del kwargs["channel_format"]
            if "num_channels" in kwargs:
                del kwargs["num_channels"]


def _check_args(kwargs: Dict[str, Any], allowed_kwargs: List[str]):
    for k, v in kwargs.items():
        if k not in allowed_kwargs:
            raise TaichiRuntimeError(
                f"Invalid argument: {k}, you can only create a graph argument with: {allowed_kwargs}"
            )
        if k == "tag":
            if not isinstance(v, ArgKind):
                raise TaichiRuntimeError(f"tag must be a ArgKind variant, but found {type(v)}.")
        if k == "name":
            if not isinstance(v, str):
                raise TaichiRuntimeError(f"name must be a string, but found {type(v)}.")


def _make_arg_scalar(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    dtype = kwargs["dtype"]
    if isinstance(dtype, MatrixType):
        raise TaichiRuntimeError(f"Tag ArgKind.SCALAR must specify a scalar type, but found {type(dtype)}.")
    return _ti_core.Arg(ArgKind.SCALAR, name, dtype, 0, [])


def _make_arg_ndarray(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
        "ndim",
        "element_shape",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    dtype = kwargs["dtype"]
    element_shape = kwargs["element_shape"]
    if isinstance(dtype, MatrixType):
        raise TaichiRuntimeError(f"Tag ArgKind.NDARRAY must specify a scalar type, but found {dtype}.")
    return _ti_core.Arg(ArgKind.NDARRAY, name, dtype, ndim, element_shape)


def _make_arg_matrix(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    dtype = kwargs["dtype"]
    if not isinstance(dtype, MatrixType):
        raise TaichiRuntimeError(f"Tag ArgKind.MATRIX must specify matrix type, but got {dtype}.")
    return _ti_core.Arg(ArgKind.MATRIX, f"{name}_mat_arg", dtype.dtype, 0, [dtype.n, dtype.m])


def _make_arg_texture(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "ndim",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    return _ti_core.Arg(ArgKind.TEXTURE, name, impl.f32, 4, [2] * ndim)


def _make_arg_rwtexture(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "ndim",
        "fmt",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    fmt = kwargs["fmt"]
    if fmt == enums.Format.unknown:
        raise TaichiRuntimeError(f"Tag ArgKind.RWTEXTURE must specify a valid color format, but found {fmt}.")
    channel_format, num_channels = FORMAT2TY_CH[fmt]
    return _ti_core.Arg(ArgKind.RWTEXTURE, name, channel_format, num_channels, [2] * ndim)


def _make_arg(kwargs: Dict[str, Any]):
    assert "tag" in kwargs
    _deprecate_arg_args(kwargs)
    proc = {
        ArgKind.SCALAR: _make_arg_scalar,
        ArgKind.NDARRAY: _make_arg_ndarray,
        ArgKind.MATRIX: _make_arg_matrix,
        ArgKind.TEXTURE: _make_arg_texture,
        ArgKind.RWTEXTURE: _make_arg_rwtexture,
    }
    tag = kwargs["tag"]
    return proc[tag](kwargs)


def _kwarg_rewriter(args, kwargs):
    for i, arg in enumerate(args):
        rewrite_map = {
            0: "tag",
            1: "name",
            2: "dtype",
            3: "ndim",
            4: "field_dim",
            5: "element_shape",
            6: "channel_format",
            7: "shape",
            8: "num_channels",
        }
        if i in rewrite_map:
            kwargs[rewrite_map[i]] = arg
        else:
            raise TaichiRuntimeError(f"Unexpected {i}th positional argument")


def Arg(*args, **kwargs):
    _kwarg_rewriter(args, kwargs)
    return _make_arg(kwargs)


__all__ = ["GraphBuilder", "Graph", "Arg", "ArgKind"]
