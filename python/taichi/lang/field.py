from functools import reduce
from operator import mul
from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.util import python_scope


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

    Each field element is a scalar, a vector, or a matrix.
    A scalar field has 1 field member. A 3x3 matrix field has 9 field members.
    A field member is a Python Expr wrapping a C++ GlobalVariableExpression.
    A C++ GlobalVariableExpression wraps the corresponding SNode.

    Args:
        vars (List[Expr]): Field members.
        tensor_shape (Tuple[Int]): Tensor shape of each field element, () if scalar.
    """
    def __init__(self, vars, tensor_shape):
        assert len(vars) == reduce(mul, tensor_shape, 1), "Tensor shape doesn't match number of vars"
        assert len(tensor_shape) <= 2, "Only scalars, vectors and matrices are supported"
        self.vars = vars
        self.tshape = tensor_shape
        self.getter = None
        self.setter = None

    @property
    def shape(self):
        """Gets field shape.

        Returns:
            Tuple[Int]: Field shape.
        """
        return self.snode.shape

    @property
    def dtype(self):
        """Gets data type of each individual value.

        Returns:
            DataType: Data type of each individual value.
        """
        return self.snode.dtype

    @property
    def tensor_shape(self):
        """Gets tensor shape of each field element.

        Returns:
            Tuple[Int]: Tensor shape of each field element, () if scalar.
        """
        return self.tshape

    @property
    def snode(self):
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        from taichi.lang.snode import SNode
        return SNode(self.vars[0].ptr.snode())

    def parent(self, n=1):
        '''XY: To be fixed:
        Create another Expr instance which represents one of the ancestors in SNode tree.

        The class it self must represent GlobalVariableExpression (field) internally.

        Args:
            n (int): levels of the target ancestor higher than the current field's snode

        Returns:
            An Expr instance which represents the target SNode ancestor internally.
        '''
        return self.snode.parent(n)

    def get_field_members(self):
        """Gets field members.

        Returns:
            List[Expr]: Field members.
        """
        return self.vars

    @python_scope
    def __setitem__(self, key, value):
        """XY: To be fixed:
        Set value with specified key when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.

        This will not be directly called from python for vector/matrix fields.
        Python Matrix class will decompose operations into scalar-level first.

        Args:
            key (Union[List[int], int, None]): indices to set
            value (Union[int, float]): value to set
        """
        self.initialize_accessor()
        self.setter(value, *self.pad_key(key))

    @python_scope
    def __getitem__(self, key):
        """XY: to fix
        Get value with specified key when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.

        This will not be directly called from python for vector/matrix fields.
        Python Matrix class will decompose operations into scalar-level first.

        Args:
            key (Union[List[int], int, None]): indices to get.

        Returns:
            Value retrieved with specified key.
        """
        self.initialize_accessor()
        return self.getter(*self.pad_key(key))

    @python_scope
    def pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key, )
        return key + ((0, ) * (_ti_core.get_max_num_indices() - len(key)))

    @python_scope
    def initialize_accessor(self):
        if self.getter:
            return
        impl.get_runtime().materialize()
        snode = self.snode.ptr
        if _ti_core.is_real(self.dtype):
            def getter(*key):
                assert len(key) == _ti_core.get_max_num_indices()
                return snode.read_float(key)

            def setter(value, *key):
                assert len(key) == _ti_core.get_max_num_indices()
                snode.write_float(key, value)
        else:
            if _ti_core.is_signed(self.dtype):
                def getter(*key):
                    assert len(key) == _ti_core.get_max_num_indices()
                    return snode.read_int(key)
            else:
                def getter(*key):
                    assert len(key) == _ti_core.get_max_num_indices()
                    return snode.read_uint(key)

            def setter(value, *key):
                assert len(key) == _ti_core.get_max_num_indices()
                snode.write_int(key, value)
        self.getter = getter
        self.setter = setter
