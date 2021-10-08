from taichi.core.util import ti_core as _ti_core
from taichi.lang.any_array import AnyArray
from taichi.lang.enums import Layout
from taichi.lang.expr import Expr
from taichi.lang.sparse_matrix import SparseMatrixBuilder
from taichi.lang.util import cook_dtype
from taichi.type.primitive_types import u64


def ext_arr():
    """Type annotation for external arrays.

    External arrays are formally defined as the data from other Python frameworks.
    For now, Taichi supports numpy and pytorch.

    Example::

        >>> @ti.kernel
        >>> def to_numpy(arr: ti.ext_arr()):
        >>>     for i in x:
        >>>         arr[i] = x[i]
        >>>
        >>> arr = numpy.zeros(...)
        >>> to_numpy(arr)  # `arr` will be filled with `x`'s data.
    """
    return ArgAnyArray()


class ArgAnyArray:
    """Type annotation for arbitrary arrays, including external arrays and Taichi ndarrays.

    For external arrays, we can treat it as a Taichi field with Vector or Matrix elements by specifying element dim and layout.
    For Taichi vector/matrix ndarrays, we will automatically identify element dim and layout. If they are explicitly specified, we will check compatibility between the actual arguments and the annotation.

    Args:
        element_dim (Union[Int, NoneType], optional): None if not specified (will be treated as 0 for external arrays), 0 if scalar elements, 1 if vector elements, and 2 if matrix elements.
        layout (Union[Layout, NoneType], optional): None if not specified (will be treated as Layout.AOS for external arrays), Layout.AOS or Layout.SOA.
    """
    def __init__(self, element_dim=None, layout=None):
        if element_dim is not None and (element_dim < 0 or element_dim > 2):
            raise ValueError(
                "Only scalars, vectors, and matrices are allowed as elements of ti.any_arr()"
            )
        self.element_dim = element_dim
        self.layout = layout

    def check_element_dim(self, arg, arg_dim):
        if self.element_dim is not None and self.element_dim != arg_dim:
            raise ValueError(
                f"Invalid argument into ti.any_arr() - required element_dim={self.element_dim}, but {arg} is provided"
            )

    def check_layout(self, arg):
        if self.layout is not None and self.layout != arg.layout:
            raise ValueError(
                f"Invalid argument into ti.any_arr() - required layout={self.layout}, but {arg} is provided"
            )


any_arr = ArgAnyArray
"""Alias for :class:`~taichi.lang.kernel_arguments.ArgAnyArray`.

Example::

    >>> @ti.kernel
    >>> def to_numpy(x: ti.any_arr(), y: ti.any_arr()):
    >>>     for i in range(n):
    >>>         x[i] = y[i]
    >>>
    >>> y = ti.ndarray(ti.f64, shape=n)
    >>> ... # calculate y
    >>> x = numpy.zeros(n)
    >>> to_numpy(x, y)  # `x` will be filled with `y`'s data.
"""


class Template:
    """Type annotation for template kernel parameter.

    See also https://docs.taichi.graphics/lang/articles/advanced/meta.

    Args:
        tensor (Any): unused
        dim (Any): unused
    """
    def __init__(self, tensor=None, dim=None):
        self.tensor = tensor
        self.dim = dim


template = Template
"""Alias for :class:`~taichi.lang.kernel_arguments.Template`.
"""


class SparseMatrixEntry:
    def __init__(self, ptr, i, j):
        self.ptr = ptr
        self.i = i
        self.j = j

    def augassign(self, value, op):
        from taichi.lang.impl import call_internal, ti_float
        if op == 'Add':
            call_internal("insert_triplet", self.ptr, self.i, self.j,
                          ti_float(value))
        elif op == 'Sub':
            call_internal("insert_triplet", self.ptr, self.i, self.j,
                          -ti_float(value))
        else:
            assert False, f"Only operations '+=' and '-=' are supported on sparse matrices."


class SparseMatrixProxy:
    def __init__(self, ptr):
        self.ptr = ptr

    def subscript(self, i, j):
        return SparseMatrixEntry(self.ptr, i, j)


sparse_matrix_builder = SparseMatrixBuilder
"""Alias for :class:`~taichi.lang.sparse_matrix.SparseMatrixBuilder`.
"""


def decl_scalar_arg(dtype):
    dtype = cook_dtype(dtype)
    arg_id = _ti_core.decl_arg(dtype, False)
    return Expr(_ti_core.make_arg_load_expr(arg_id, dtype))


def decl_sparse_matrix():
    ptr_type = cook_dtype(u64)
    # Treat the sparse matrix argument as a scalar since we only need to pass in the base pointer
    arg_id = _ti_core.decl_arg(ptr_type, False)
    return SparseMatrixProxy(_ti_core.make_arg_load_expr(arg_id, ptr_type))


def decl_any_arr_arg(dtype, dim, element_shape, layout):
    dtype = cook_dtype(dtype)
    arg_id = _ti_core.decl_arg(dtype, True)
    element_dim = len(element_shape)
    if layout == Layout.AOS:
        element_dim = -element_dim
    return AnyArray(
        _ti_core.make_external_tensor_expr(dtype, dim, arg_id, element_dim),
        element_shape, layout)


def decl_scalar_ret(dtype):
    dtype = cook_dtype(dtype)
    return _ti_core.decl_ret(dtype)
