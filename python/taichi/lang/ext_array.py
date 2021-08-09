from taichi.core.util import ti_core as _ti_core
from taichi.lang.expr import Expr
from taichi.lang.util import taichi_scope


class ExtArray:
    """Class for external arrays. Constructed by a C++ Expr wrapping a C++ ExternalTensorExpression.

    Args:
        ptr: The C++ Expr mentioned above.
    """
    def __init__(self, ptr):
        assert ptr.is_external_var()
        self.ptr = ptr

    @property
    @taichi_scope
    def shape(self):
        """A list containing sizes for each dimension.

        Returns:
            List[Int]: The result list.
        """
        dim = _ti_core.get_external_tensor_dim(self.ptr)
        ret = [
            Expr(_ti_core.get_external_tensor_shape_along_axis(self.ptr, i))
            for i in range(dim)
        ]
        return ret

    @taichi_scope
    def loop_range(self):
        """Gets the corresponding C++ Expr to serve as loop range.

        Returns:
            C++ Expr: See above.
        """
        return self.ptr
