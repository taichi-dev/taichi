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
        assert ptr.is_external_tensor_expr()
        self.ptr = ptr
        self.ptr.type_check(impl.get_runtime().prog.config())

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
            _ti_core.get_external_tensor_element_type(self.ptr), None, _ti_core.get_external_tensor_needs_grad(self.ptr)
        )  # AnyArray can take any shape

    @property
    @taichi_scope
    def grad(self):
        """Returns the gradient of this array."""
        return AnyArray(_ti_core.make_external_tensor_grad_expr(self.ptr))

    @property
    @taichi_scope
    def shape(self):
        """A list containing sizes for each dimension. Note that element shape will be excluded.

        Returns:
            List[Int]: The result list.
        """
        dim = _ti_core.get_external_tensor_dim(self.ptr)
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        return [Expr(_ti_core.get_external_tensor_shape_along_axis(self.ptr, i, dbg_info)) for i in range(dim)]

    @taichi_scope
    def _loop_range(self):
        """Gets the corresponding taichi_python.Expr to serve as loop range.

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
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()

        indices_second = (i,) if len(self.arr.element_shape()) == 1 else (i, j)
        if self.arr.layout() == Layout.SOA:
            indices = indices_second + self.indices_first
        else:
            indices = self.indices_first + indices_second
        return Expr(
            ast_builder.expr_subscript(
                self.arr.ptr,
                make_expr_group(*indices),
                _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )
        )


__all__ = []
