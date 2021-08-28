from taichi.core.primitive_types import u64
from taichi.core.util import ti_core as _ti_core
from taichi.lang.enums import Layout
from taichi.lang.expr import Expr
from taichi.lang.ext_array import AnyArray, ExtArray
from taichi.lang.snode import SNode
from taichi.lang.sparse_matrix import SparseMatrixBuilder
from taichi.lang.util import cook_dtype, to_taichi_type


class ArgExtArray:
    """Type annotation for external arrays.

    External array is formally defined as the data from other Python frameworks.
    For now, Taichi supports numpy and pytorch.

    Args:
        dim (int, optional): must be 1.
    """
    def __init__(self, dim=1):
        assert dim == 1

    def extract(self, x):
        return to_taichi_type(x.dtype), len(x.shape)


ext_arr = ArgExtArray
"""Alias for :class:`~taichi.lang.kernel_arguments.ArgExtArray`.

Example::

    >>> @ti.kernel
    >>> def to_numpy(arr: ti.ext_arr()):
    >>>     for i in x:
    >>>         arr[i] = x[i]
    >>>
    >>> arr = numpy.zeros(...)
    >>> to_numpy(arr)  # `arr` will be filled with `x`'s data.
"""


class ArgAnyArray:
    """Type annotation for arbitrary arrays, including external arrays and Taichi ndarrays.

    For external arrays, we can treat it as a Taichi field with Vector or Matrix elements by specifying element shape and layout.

    Args:
        element_shape (Tuple[Int], optional): () if scalar elements (default), (n) if vector elements, and (n, m) if matrix elements.
        layout (Layout, optional): Memory layout, AOS by default.
    """
    def __init__(self, element_shape=(), layout=Layout.AOS):
        if len(element_shape) > 2:
            raise ValueError(
                "Only scalars, vectors, and matrices are allowed as elements of ti.any_arr()"
            )
        self.element_shape = element_shape
        self.layout = layout

    def extract(self, x):
        shape = tuple(x.shape)
        element_dim = len(self.element_shape)
        if len(shape) < element_dim:
            raise ValueError("Invalid argument passed to ti.any_arr()")
        if element_dim > 0:
            if self.layout == Layout.SOA:
                if shape[:element_dim] != self.element_shape:
                    raise ValueError("Invalid argument passed to ti.any_arr()")
            else:
                if shape[-element_dim:] != self.element_shape:
                    raise ValueError("Invalid argument passed to ti.any_arr()")
        return to_taichi_type(
            x.dtype), len(shape), self.element_shape, self.layout


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

    See also https://docs.taichi.graphics/docs/lang/articles/advanced/meta.

    Args:
        tensor (Any): unused
        dim (Any): unused
    """
    def __init__(self, tensor=None, dim=None):
        self.tensor = tensor
        self.dim = dim

    def extract(self, x):
        if isinstance(x, SNode):
            return x.ptr
        if isinstance(x, Expr):
            return x.ptr.get_underlying_ptr_address()
        if isinstance(x, _ti_core.Expr):
            return x.get_underlying_ptr_address()
        if isinstance(x, tuple):
            return tuple(self.extract(item) for item in x)
        return x


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
    is_taichi_class = True

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


def decl_ext_arr_arg(dtype, dim):
    dtype = cook_dtype(dtype)
    arg_id = _ti_core.decl_arg(dtype, True)
    return ExtArray(_ti_core.make_external_tensor_expr(dtype, dim, arg_id))


def decl_sparse_matrix():
    ptr_type = cook_dtype(u64)
    # Treat the sparse matrix argument as a scalar since we only need to pass in the base pointer
    arg_id = _ti_core.decl_arg(ptr_type, False)
    return SparseMatrixProxy(_ti_core.make_arg_load_expr(arg_id, ptr_type))


def decl_any_arr_arg(dtype, dim, element_shape, layout):
    dtype = cook_dtype(dtype)
    arg_id = _ti_core.decl_arg(dtype, True)
    return AnyArray(_ti_core.make_external_tensor_expr(dtype, dim, arg_id),
                    element_shape, layout)


def decl_scalar_ret(dtype):
    dtype = cook_dtype(dtype)
    return _ti_core.decl_ret(dtype)
