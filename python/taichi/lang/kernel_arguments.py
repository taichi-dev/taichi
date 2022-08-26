import inspect

import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang import impl, ops
from taichi.lang._texture import RWTextureAccessor, TextureSampler
from taichi.lang.any_array import AnyArray
from taichi.lang.enums import Layout
from taichi.lang.expr import Expr
from taichi.lang.matrix import Matrix, MatrixType
from taichi.lang.util import cook_dtype
from taichi.types.primitive_types import RefType, f32, u64


class KernelArgument:
    def __init__(self, _annotation, _name, _default=inspect.Parameter.empty):
        self.annotation = _annotation
        self.name = _name
        self.default = _default


class SparseMatrixEntry:
    def __init__(self, ptr, i, j, dtype):
        self.ptr = ptr
        self.i = i
        self.j = j
        self.dtype = dtype

    def _augassign(self, value, op):
        call_func = f"insert_triplet_{self.dtype}"
        if op == 'Add':
            taichi.lang.impl.call_internal(call_func, self.ptr, self.i, self.j,
                                           ops.cast(value, self.dtype))
        elif op == 'Sub':
            taichi.lang.impl.call_internal(call_func, self.ptr, self.i, self.j,
                                           -ops.cast(value, self.dtype))
        else:
            assert False, "Only operations '+=' and '-=' are supported on sparse matrices."


class SparseMatrixProxy:
    def __init__(self, ptr, dtype):
        self.ptr = ptr
        self.dtype = dtype

    def subscript(self, i, j):
        return SparseMatrixEntry(self.ptr, i, j, self.dtype)


def decl_scalar_arg(dtype):
    is_ref = False
    if isinstance(dtype, RefType):
        is_ref = True
        dtype = dtype.tp
    dtype = cook_dtype(dtype)
    arg_id = impl.get_runtime().prog.decl_scalar_arg(dtype)
    return Expr(_ti_core.make_arg_load_expr(arg_id, dtype, is_ref))


def decl_matrix_arg(matrixtype):
    return Matrix(
        [[decl_scalar_arg(matrixtype.dtype) for _ in range(matrixtype.m)]
         for _ in range(matrixtype.n)],
        ndim=matrixtype.ndim)


def decl_sparse_matrix(dtype):
    value_type = cook_dtype(dtype)
    ptr_type = cook_dtype(u64)
    # Treat the sparse matrix argument as a scalar since we only need to pass in the base pointer
    arg_id = impl.get_runtime().prog.decl_scalar_arg(ptr_type)
    return SparseMatrixProxy(
        _ti_core.make_arg_load_expr(arg_id, ptr_type, False), value_type)


def decl_ndarray_arg(dtype, dim, element_shape, layout):
    dtype = cook_dtype(dtype)
    element_dim = len(element_shape)
    arg_id = impl.get_runtime().prog.decl_arr_arg(dtype, dim, element_shape)
    if layout == Layout.AOS:
        element_dim = -element_dim
    return AnyArray(
        _ti_core.make_external_tensor_expr(dtype, dim, arg_id, element_dim,
                                           element_shape))


def decl_texture_arg(num_dimensions):
    # FIXME: texture_arg doesn't have element_shape so better separate them
    arg_id = impl.get_runtime().prog.decl_texture_arg(f32)
    return TextureSampler(
        _ti_core.make_texture_ptr_expr(arg_id, num_dimensions), num_dimensions)


def decl_rw_texture_arg(num_dimensions, num_channels, channel_format, lod):
    # FIXME: texture_arg doesn't have element_shape so better separate them
    arg_id = impl.get_runtime().prog.decl_texture_arg(f32)
    return RWTextureAccessor(
        _ti_core.make_rw_texture_ptr_expr(arg_id, num_dimensions, num_channels,
                                          channel_format, lod), num_dimensions)


def decl_ret(dtype):
    if isinstance(dtype, MatrixType):
        dtype = _ti_core.get_type_factory_instance().get_tensor_type(
            [dtype.n, dtype.m], dtype.dtype)
    else:
        dtype = cook_dtype(dtype)
    return impl.get_runtime().prog.decl_ret(dtype)
