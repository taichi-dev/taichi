from typing import Any

from taichi.lang._ndarray import ScalarNdarray
from taichi.lang._texture import Texture
from taichi.lang.enums import Format
from taichi.lang.exception import TaichiCompilationError
from taichi.lang.matrix import (
    Matrix,
    MatrixNdarray,
    MatrixType,
    VectorNdarray,
    VectorType,
)
from taichi.lang.util import cook_dtype
from taichi.types.annotations import template
from taichi.types.ndarray_type import NdarrayType
from taichi.types.texture_type import RWTextureType, TextureType

template_types = (NdarrayType, TextureType, template)


def check_type_match(lhs, rhs):
    if isinstance(lhs, MatrixType) and isinstance(rhs, MatrixType):
        return lhs.n == rhs.n and lhs.m == rhs.m and (lhs.dtype == rhs.dtype or lhs.dtype is None or rhs.dtype is None)
    if isinstance(lhs, MatrixType) or isinstance(rhs, MatrixType):
        return False

    return cook_dtype(lhs) == cook_dtype(rhs)


def produce_injected_args_from_template(kernel, template_args):
    injected_args = []
    num_template_args = len([arg.annotation for arg in kernel.arguments if isinstance(arg.annotation, template_types)])
    assert num_template_args == len(
        template_args
    ), f"Need {num_template_args} inputs to instantiate the template parameters, got {len(template_args)}"
    for arg in kernel.arguments:
        anno = arg.annotation
        if isinstance(anno, template_types):
            injected_args.append(template_args[arg.name])
        elif isinstance(anno, RWTextureType):
            texture_shape = (2,) * anno.num_dimensions
            fmt = anno.fmt
            injected_args.append(Texture(fmt, texture_shape))
        else:
            injected_args.append(0)
    return injected_args


def produce_injected_args(kernel, symbolic_args=None):
    injected_args = []
    for i, arg in enumerate(kernel.arguments):
        anno = arg.annotation
        if isinstance(anno, NdarrayType):
            if symbolic_args is not None:
                # TODO: reconstruct dtype to be TensorType from taichi_core instead of the Python ones
                element_dim = len(symbolic_args[i].element_shape)
                if element_dim == 0 or symbolic_args[i].element_shape == (1,):
                    dtype = symbolic_args[i].dtype()
                elif element_dim == 1:
                    dtype = VectorType(symbolic_args[i].element_shape[0], symbolic_args[i].dtype())
                elif element_dim == 2:
                    dtype = MatrixType(
                        symbolic_args[i].element_shape[0],
                        symbolic_args[i].element_shape[1],
                        2,
                        symbolic_args[i].dtype(),
                    )
                else:
                    raise TaichiCompilationError("Not supported")
                ndim = symbolic_args[i].field_dim
            else:
                ndim = anno.ndim
                dtype = anno.dtype

            if anno.ndim is not None and ndim != anno.ndim:
                raise TaichiCompilationError(
                    f"{ndim} from Arg {arg.name} doesn't match kernel's annotated ndim={anno.ndim}"
                )

            if anno.dtype is not None and not check_type_match(dtype, anno.dtype):
                raise TaichiCompilationError(
                    f" Arg {arg.name}'s dtype {dtype.to_string()} doesn't match kernel's annotated dtype={anno.dtype.to_string()}"
                )

            if isinstance(dtype, VectorType):
                injected_args.append(VectorNdarray(dtype.n, dtype=dtype.dtype, shape=(2,) * ndim))
            elif isinstance(dtype, MatrixType):
                injected_args.append(MatrixNdarray(dtype.n, dtype.m, dtype=dtype.dtype, shape=(2,) * ndim))
            else:
                injected_args.append(ScalarNdarray(dtype, (2,) * ndim))
        elif isinstance(anno, RWTextureType):
            texture_shape = (2,) * anno.num_dimensions
            fmt = anno.fmt
            injected_args.append(Texture(fmt, texture_shape))
        elif isinstance(anno, TextureType):
            texture_shape = (2,) * anno.num_dimensions
            injected_args.append(Texture(Format.rgba8, texture_shape))
        elif isinstance(anno, MatrixType):
            if symbolic_args is not None:
                symbolic_mat_n = symbolic_args[i].element_shape[0]
                symbolic_mat_m = symbolic_args[i].element_shape[1]

                if symbolic_mat_m != anno.m or symbolic_mat_n != anno.n:
                    raise RuntimeError(
                        f"Matrix dimension mismatch, expected ({anno.n}, {anno.m}) "
                        f"but dispatched shape ({symbolic_mat_n}, {symbolic_mat_m})."
                    )
            injected_args.append(Matrix([0] * anno.n * anno.m, dt=anno.dtype))
        else:
            if symbolic_args is not None:
                dtype = symbolic_args[i].dtype()
            else:
                dtype = anno

            if not check_type_match(dtype, anno):
                raise TaichiCompilationError(
                    f" Arg {arg.name}'s dtype {dtype.to_string()} doesn't match kernel's annotated dtype={anno.to_string()}"
                )
            # For primitive types, we can just inject a dummy value.
            injected_args.append(0)
    return injected_args


def json_data_model(f):
    """
    Decorates a JSON data model. A JSON data model MUST NOT have any member
    functions and it MUST be constructible from a JSON object.

    This is merely a marker.
    """
    f._is_json_data_model = True
    return f


def is_json_data_model(cls) -> bool:
    return hasattr(cls, "_is_json_data_model")


def dump_json_data_model(x: object) -> Any:
    if isinstance(x, (int, float, str, bool, type(None))):
        return x
    if isinstance(x, (list, tuple)):
        return [dump_json_data_model(e) for e in x]
    if isinstance(x, dict):
        return {k: dump_json_data_model(v) for k, v in x.items()}
    if is_json_data_model(x):
        return {k: dump_json_data_model(v) for k, v in x.__dict__.items() if k != "_is_json_data_model"}
    return x
