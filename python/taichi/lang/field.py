from functools import reduce
from operator import mul

class Field:
    """Taichi field abstract class."""

    @property
    def shape(self):
        raise Exception("Abstract Field class should not be directly used")

    @property
    def dtype(self):
        raise Exception("Abstract Field class should not be directly used")

    @property
    def tensor_shape(self):
        raise Exception("Abstract Field class should not be directly used")

    @property
    def is_tensor(self):
        return len(self.tensor_shape) > 0


class SNodeField(Field):
    """Taichi field with SNode implementation.

    Args:
        vars (List[Expr]): Field members wrapping corresponding C++ GlobalVariableExpressions.
        tensor_shape (tuple): Tensor shape, () if scalar.
    """
    def __init__(self, vars, tensor_shape):
        assert len(vars) == reduce(mul, tensor_shape, 1), "Tensor shape doesn't match number of vars"
        assert len(tensor_shape) <= 2, "Only scalars, vectors and matrices are supported"
        self.vars = vars
        self.tshape = tensor_shape

    @property
    def shape(self):
        raise self.snode.shape

    @property
    def dtype(self):
        return self.snode.dtype

    @property
    def tensor_shape(self):
        return self.tshape

    @property
    def snode(self):
        from taichi.lang.snode import SNode
        return SNode(self.vars[0].ptr.snode())

    def get_field_members(self):
        return self.vars
