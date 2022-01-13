import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang.any_array import AnyArray
from taichi.lang.enums import Layout
from taichi.lang.expr import Expr
from taichi.lang.matrix import Matrix
from taichi.lang.util import cook_dtype
from taichi.types.primitive_types import u64


class SparseMatrixEntry:
    def __init__(self, ptr, i, j):
        self.ptr = ptr
        self.i = i
        self.j = j

    def augassign(self, value, op):
        if op == 'Add':
            taichi.lang.impl.call_internal("insert_triplet", self.ptr, self.i,
                                           self.j,
                                           taichi.lang.impl.ti_float(value))
        elif op == 'Sub':
            taichi.lang.impl.call_internal("insert_triplet", self.ptr, self.i,
                                           self.j,
                                           -taichi.lang.impl.ti_float(value))
        else:
            assert False, "Only operations '+=' and '-=' are supported on sparse matrices."


class SparseMatrixProxy:
    def __init__(self, ptr):
        self.ptr = ptr

    def subscript(self, i, j):
        return SparseMatrixEntry(self.ptr, i, j)


def decl_scalar_arg(dtype):
    dtype = cook_dtype(dtype)
    arg_id = _ti_core.decl_arg(dtype, False)
    return Expr(_ti_core.make_arg_load_expr(arg_id, dtype))


def decl_matrix_arg(matrixtype):
    return Matrix(
        [[decl_scalar_arg(matrixtype.dtype) for _ in range(matrixtype.m)]
         for _ in range(matrixtype.n)])


def decl_sparse_matrix():
    ptr_type = cook_dtype(u64)
    # Treat the sparse matrix argument as a scalar since we only need to pass in the base pointer
    arg_id = _ti_core.decl_arg(ptr_type, False)
    return SparseMatrixProxy(_ti_core.make_arg_load_expr(arg_id, ptr_type))


def decl_any_arr_arg(dtype, dim, element_shape, layout):
    dtype = cook_dtype(dtype)
    element_dim = len(element_shape)
    arg_id = _ti_core.decl_arr_arg(dtype, dim, element_shape)
    if layout == Layout.AOS:
        element_dim = -element_dim
    return AnyArray(
        _ti_core.make_external_tensor_expr(dtype, dim, arg_id, element_dim),
        element_shape, layout)


def decl_scalar_ret(dtype):
    dtype = cook_dtype(dtype)
    return _ti_core.decl_ret(dtype)
