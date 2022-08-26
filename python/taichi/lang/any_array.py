from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.enums import Layout
from taichi.lang.expr import Expr, make_expr_group
from taichi.lang.util import taichi_scope
from taichi.types.ndarray_type import NdarrayTypeMetadata


class AnyArray:
    """Class for arbitrary arrays in Python AST.

    Args:
        ptr (taichi_python.Expr): A taichi_python.Expr wrapping a taichi_python.ExternalTensorExpression.
        element_shape (Tuple[Int]): () if scalar elements (default), (n) if vector elements, and (n, m) if matrix elements.
        layout (Layout): Memory layout.
    """
    def __init__(self, ptr):
        assert ptr.is_external_var()
        self.ptr = ptr

    def element_shape(self):
        return _ti_core.get_external_tensor_element_shape(self.ptr)

    def layout(self):
        # 0: scalar; 1: vector (SOA); 2: matrix (SOA); -1: vector
        # (AOS); -2: matrix (AOS)
        element_dim = _ti_core.get_external_tensor_element_dim(self.ptr)
        if element_dim == 1 or element_dim == 2:
            return Layout.SOA
        return Layout.AOS

    def get_type(self):
        return NdarrayTypeMetadata(
            self.ptr.get_ret_type(),
            None,  # AnyArray can take any shape
            self.layout())

    @property
    @taichi_scope
    def shape(self):
        """A list containing sizes for each dimension. Note that element shape will be excluded.

        Returns:
            List[Int]: The result list.
        """
        dim = _ti_core.get_external_tensor_dim(self.ptr)
        ret = [
            Expr(_ti_core.get_external_tensor_shape_along_axis(self.ptr, i))
            for i in range(dim)
        ]
        element_dim = len(self.element_shape())
        if element_dim == 0:
            return ret
        return ret[element_dim:] if self.layout(
        ) == Layout.SOA else ret[:-element_dim]

    @taichi_scope
    def _loop_range(self):
        """Gets the corresponding taichi_python.Expr to serve as loop range.

        This is not in use now because struct fors on AnyArrays are not supported yet.

        Returns:
            taichi_python.Expr: See above.
        """
        return self.ptr


class AnyArrayAccess:
    """Class for first-level access to AnyArray with Vector/Matrix elements in Python AST.

    Args:
        arr (AnyArray): See above.
        indices_first (Tuple[Int]): Indices of first-level access.
    """
    def __init__(self, arr, indices_first):
        self.arr = arr
        self.indices_first = indices_first

    @taichi_scope
    def subscript(self, i, j):
        indices_second = (i, ) if len(self.arr.element_shape()) == 1 else (i,
                                                                           j)
        if self.arr.layout() == Layout.SOA:
            indices = indices_second + self.indices_first
        else:
            indices = self.indices_first + indices_second
        return Expr(
            _ti_core.subscript(self.arr.ptr, make_expr_group(*indices),
                               impl.get_runtime().get_current_src_info()))


__all__ = []
