import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang.util import python_scope, to_numpy_type, to_pytorch_type


class Field:
    """Taichi field with SNode implementation.

    A field is constructed by a list of field members.
    For example, a scalar field has 1 field member, while a 3x3 matrix field has 9 field members.
    A field member is a Python Expr wrapping a C++ GlobalVariableExpression.
    A C++ GlobalVariableExpression wraps the corresponding SNode.

    Args:
        vars (List[Expr]): Field members.
    """
    def __init__(self, _vars):
        self.vars = _vars
        self.host_accessors = None
        self.grad = None

    @property
    def snode(self):
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        return self._snode

    @property
    def _snode(self):
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        return taichi.lang.snode.SNode(self.vars[0].ptr.snode())

    @property
    def shape(self):
        """Gets field shape.

        Returns:
            Tuple[Int]: Field shape.
        """
        return self._snode.shape

    @property
    def dtype(self):
        """Gets data type of each individual value.

        Returns:
            DataType: Data type of each individual value.
        """
        return self._snode._dtype

    @property
    def _name(self):
        """Gets field name.

        Returns:
            str: Field name.
        """
        return self._snode._name

    def parent(self, n=1):
        """Gets an ancestor of the representative SNode in the SNode tree.

        Args:
            n (int): the number of levels going up from the representative SNode.

        Returns:
            SNode: The n-th parent of the representative SNode.
        """
        return self.snode.parent(n)

    def _get_field_members(self):
        """Gets field members.

        Returns:
            List[Expr]: Field members.
        """
        return self.vars

    def _loop_range(self):
        """Gets representative field member for loop range info.

        Returns:
            taichi_core.Expr: Representative (first) field member.
        """
        return self.vars[0].ptr

    def _set_grad(self, grad):
        """Sets corresponding gradient field.

        Args:
            grad (Field): Corresponding gradient field.
        """
        self.grad = grad

    @python_scope
    def fill(self, val):
        """Fills `self` with a specific value.

        Args:
            val (Union[int, float]): Value to fill.
        """
        raise NotImplementedError()

    @python_scope
    def to_numpy(self, dtype=None):
        """Converts `self` to a numpy array.

        Args:
            dtype (DataType, optional): The desired data type of returned numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        raise NotImplementedError()

    @python_scope
    def to_torch(self, device=None):
        """Converts `self` to a torch tensor.

        Args:
            device (torch.device, optional): The desired device of returned tensor.

        Returns:
            torch.tensor: The result torch tensor.
        """
        raise NotImplementedError()

    @python_scope
    def from_numpy(self, arr):
        """Loads all elements from a numpy array.

        The shape of the numpy array needs to be the same as `self`.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        raise NotImplementedError()

    @python_scope
    def from_torch(self, arr):
        """Loads all elements from a torch tensor.

        The shape of the torch tensor needs to be the same as `self`.

        Args:
            arr (torch.tensor): The source torch tensor.
        """
        self.from_numpy(arr.contiguous())

    @python_scope
    def copy_from(self, other):
        """Copies all elements from another field.

        The shape of the other field needs to be the same as `self`.

        Args:
            other (Field): The source field.
        """
        if not isinstance(other, Field):
            raise TypeError('Cannot copy from a non-field object')
        if self.shape != other.shape:
            raise ValueError(f"ti.field shape {self.shape} does not match"
                             f" the source field shape {other.shape}")
        from taichi._kernels import tensor_to_tensor  # pylint: disable=C0415
        tensor_to_tensor(self, other)

    @python_scope
    def __setitem__(self, key, value):
        """Sets field element in Python scope.

        Args:
            key (Union[List[int], int, None]): Coordinates of the field element.
            value (element type): Value to set.
        """
        raise NotImplementedError()

    @python_scope
    def __getitem__(self, key):
        """Gets field element in Python scope.

        Args:
            key (Union[List[int], int, None]): Coordinates of the field element.

        Returns:
            element type: Value retrieved.
        """
        raise NotImplementedError()

    def __str__(self):
        if taichi.lang.impl.inside_kernel():
            return self.__repr__()  # make pybind11 happy, see Matrix.__str__
        if self._snode.ptr is None:
            return '<Field: Definition of this field is incomplete>'
        return str(self.to_numpy())

    def _pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key, )
        assert len(key) == len(self.shape)
        return key + ((0, ) * (_ti_core.get_max_num_indices() - len(key)))

    def _initialize_host_accessors(self):
        if self.host_accessors:
            return
        taichi.lang.impl.get_runtime().materialize()
        self.host_accessors = [
            SNodeHostAccessor(e.ptr.snode()) for e in self.vars
        ]

    def _host_access(self, key):
        return [SNodeHostAccess(e, key) for e in self.host_accessors]


class ScalarField(Field):
    """Taichi scalar field with SNode implementation.

    Args:
        var (Expr): Field member.
    """
    def __init__(self, var):
        super().__init__([var])

    @python_scope
    def fill(self, val):
        from taichi._kernels import fill_tensor  # pylint: disable=C0415
        fill_tensor(self, val)

    @python_scope
    def to_numpy(self, dtype=None):
        if dtype is None:
            dtype = to_numpy_type(self.dtype)
        import numpy as np  # pylint: disable=C0415
        arr = np.zeros(shape=self.shape, dtype=dtype)
        from taichi._kernels import tensor_to_ext_arr  # pylint: disable=C0415
        tensor_to_ext_arr(self, arr)
        taichi.lang.runtime_ops.sync()
        return arr

    @python_scope
    def to_torch(self, device=None):
        import torch  # pylint: disable=C0415

        # pylint: disable=E1101
        arr = torch.zeros(size=self.shape,
                          dtype=to_pytorch_type(self.dtype),
                          device=device)
        from taichi._kernels import tensor_to_ext_arr  # pylint: disable=C0415
        tensor_to_ext_arr(self, arr)
        taichi.lang.runtime_ops.sync()
        return arr

    @python_scope
    def from_numpy(self, arr):
        if len(self.shape) != len(arr.shape):
            raise ValueError(f"ti.field shape {self.shape} does not match"
                             f" the numpy array shape {arr.shape}")
        for i, _ in enumerate(self.shape):
            if self.shape[i] != arr.shape[i]:
                raise ValueError(f"ti.field shape {self.shape} does not match"
                                 f" the numpy array shape {arr.shape}")
        if hasattr(arr, 'contiguous'):
            arr = arr.contiguous()
        from taichi._kernels import ext_arr_to_tensor  # pylint: disable=C0415
        ext_arr_to_tensor(arr, self)
        taichi.lang.runtime_ops.sync()

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessors()
        self.host_accessors[0].setter(value, *self._pad_key(key))

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessors()
        return self.host_accessors[0].getter(*self._pad_key(key))

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        return '<ti.field>'


class SNodeHostAccessor:
    def __init__(self, snode):
        if _ti_core.is_real(snode.data_type()):

            def getter(*key):
                assert len(key) == _ti_core.get_max_num_indices()
                return snode.read_float(key)

            def setter(value, *key):
                assert len(key) == _ti_core.get_max_num_indices()
                snode.write_float(key, value)
        else:
            if _ti_core.is_signed(snode.data_type()):

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


class SNodeHostAccess:
    def __init__(self, accessor, key):
        self.accessor = accessor
        self.key = key


__all__ = ["Field", "ScalarField"]
