import numbers

from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.util import python_scope, to_numpy_type, to_pytorch_type
from taichi.misc.util import warning

import taichi as ti


class Field:
    """Taichi field with SNode implementation.

    A field is constructed by a list of field members.
    For example, a scalar field has 1 field member, while a 3x3 matrix field has 9 field members.
    A field member is a Python Expr wrapping a C++ GlobalVariableExpression.
    A C++ GlobalVariableExpression wraps the corresponding SNode.

    Args:
        vars (List[Expr]): Field members.
    """
    def __init__(self, vars):
        self.vars = vars
        self.host_accessors = None
        self.grad = None

    @property
    def snode(self):
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        from taichi.lang.snode import SNode
        return SNode(self.vars[0].ptr.snode())

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
    def name(self):
        """Gets field name.

        Returns:
            str: Field name.
        """
        return self.snode.name

    def parent(self, n=1):
        """Gets an ancestor of the representative SNode in the SNode tree.

        Args:
            n (int): the number of levels going up from the representative SNode.

        Returns:
            SNode: The n-th parent of the representative SNode.
        """
        return self.snode.parent(n)

    def get_field_members(self):
        """Gets field members.

        Returns:
            List[Expr]: Field members.
        """
        return self.vars

    def loop_range(self):
        """Gets representative field member for loop range info.

        Returns:
            Expr: Representative (first) field member.
        """
        return self.vars[0]

    def set_grad(self, grad):
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
        assert isinstance(other, Field)
        from taichi.lang.meta import tensor_to_tensor
        assert len(self.shape) == len(other.shape)
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
        if impl.inside_kernel():
            return self.__repr__()  # make pybind11 happy, see Matrix.__str__
        else:
            return str(self.to_numpy())

    def pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key, )
        assert len(key) == len(self.shape)
        return key + ((0, ) * (_ti_core.get_max_num_indices() - len(key)))

    def initialize_host_accessors(self):
        if self.host_accessors:
            return
        impl.get_runtime().materialize()
        self.host_accessors = [
            SNodeHostAccessor(e.ptr.snode()) for e in self.vars
        ]


class ScalarField(Field):
    """Taichi scalar field with SNode implementation.

    Args:
        var (Expr): Field member.
    """
    def __init__(self, var):
        super().__init__([var])

    @python_scope
    def fill(self, val):
        from taichi.lang.meta import fill_tensor
        fill_tensor(self, val)

    @python_scope
    def to_numpy(self, dtype=None):
        if dtype is None:
            dtype = to_numpy_type(self.dtype)
        import numpy as np
        arr = np.zeros(shape=self.shape, dtype=dtype)
        from taichi.lang.meta import tensor_to_ext_arr
        tensor_to_ext_arr(self, arr)
        ti.sync()
        return arr

    @python_scope
    def to_torch(self, device=None):
        import torch
        arr = torch.zeros(size=self.shape,
                          dtype=to_pytorch_type(self.dtype),
                          device=device)
        from taichi.lang.meta import tensor_to_ext_arr
        tensor_to_ext_arr(self, arr)
        ti.sync()
        return arr

    @python_scope
    def from_numpy(self, arr):
        assert len(self.shape) == len(arr.shape)
        for i in range(len(self.shape)):
            assert self.shape[i] == arr.shape[i]
        if hasattr(arr, 'contiguous'):
            arr = arr.contiguous()
        from taichi.lang.meta import ext_arr_to_tensor
        ext_arr_to_tensor(arr, self)
        ti.sync()

    @python_scope
    def __setitem__(self, key, value):
        self.initialize_host_accessors()
        self.host_accessors[0].setter(value, *self.pad_key(key))

    @python_scope
    def __getitem__(self, key):
        self.initialize_host_accessors()
        return self.host_accessors[0].getter(*self.pad_key(key))

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        return '<ti.field>'


class MatrixField(Field):
    """Taichi matrix field with SNode implementation.

    Args:
        vars (Expr): Field members.
        n (Int): Number of rows.
        m (Int): Number of columns.
    """
    def __init__(self, vars, n, m):
        assert len(vars) == n * m
        super().__init__(vars)
        self.n = n
        self.m = m

    def get_scalar_field(self, *indices):
        """Creates a ScalarField using a specific field member. Only used for quant.

        Args:
            indices (Tuple[Int]): Specified indices of the field member.

        Returns:
            ScalarField: The result ScalarField.
        """
        assert len(indices) in [1, 2]
        i = indices[0]
        j = 0 if len(indices) == 1 else indices[1]
        return ScalarField(self.vars[i * self.m + j])

    @python_scope
    def fill(self, val):
        """Fills `self` with specific values.

        Args:
            val (Union[Number, List, Tuple, Matrix]): Values to fill, which should have dimension consistent with `self`.
        """
        if isinstance(val, numbers.Number):
            val = tuple(
                [tuple([val for _ in range(self.m)]) for _ in range(self.n)])
        elif isinstance(val,
                        (list, tuple)) and isinstance(val[0], numbers.Number):
            assert self.m == 1
            val = tuple([(v, ) for v in val])
        elif isinstance(val, ti.Matrix):
            val_tuple = []
            for i in range(val.n):
                row = []
                for j in range(val.m):
                    row.append(val(i, j))
                row = tuple(row)
                val_tuple.append(row)
            val = tuple(val_tuple)
        assert len(val) == self.n
        assert len(val[0]) == self.m
        from taichi.lang.meta import fill_matrix
        fill_matrix(self, val)

    @python_scope
    def to_numpy(self, keep_dims=False, as_vector=None, dtype=None):
        """Converts `self` to a numpy array.

        Args:
            keep_dims (bool, optional): Whether to keep the dimension after conversion.
                When keep_dims=True, on an n-D matrix field, the numpy array always has n+2 dims, even for 1x1, 1xn, nx1 matrix fields.
                When keep_dims=False, the resulting numpy array should skip the matrix dims with size 1.
                For example, a 4x1 or 1x4 matrix field with 5x6x7 elements results in an array of shape 5x6x7x4.
            as_vector (bool, deprecated): Whether to make the returned numpy array as a vector, i.e., with shape (n,) rather than (n, 1).
                Note that this argument has been deprecated.
                More discussion about `as_vector`: https://github.com/taichi-dev/taichi/pull/1046#issuecomment-633548858.
            dtype (DataType, optional): The desired data type of returned numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        if as_vector is not None:
            warning(
                'v.to_numpy(as_vector=True) is deprecated, '
                'please use v.to_numpy() directly instead',
                DeprecationWarning,
                stacklevel=3)
        if dtype is None:
            dtype = to_numpy_type(self.dtype)
        as_vector = self.m == 1 and not keep_dims
        shape_ext = (self.n, ) if as_vector else (self.n, self.m)
        import numpy as np
        arr = np.zeros(self.shape + shape_ext, dtype=dtype)
        from taichi.lang.meta import matrix_to_ext_arr
        matrix_to_ext_arr(self, arr, as_vector)
        ti.sync()
        return arr

    def to_torch(self, device=None, keep_dims=False):
        """Converts `self` to a torch tensor.

        Args:
            device (torch.device, optional): The desired device of returned tensor.
            keep_dims (bool, optional): Whether to keep the dimension after conversion.
                See :meth:`~taichi.lang.field.MatrixField.to_numpy` for more detailed explanation.

        Returns:
            torch.tensor: The result torch tensor.
        """
        import torch
        as_vector = self.m == 1 and not keep_dims
        shape_ext = (self.n, ) if as_vector else (self.n, self.m)
        arr = torch.empty(self.shape + shape_ext,
                          dtype=to_pytorch_type(self.dtype),
                          device=device)
        from taichi.lang.meta import matrix_to_ext_arr
        matrix_to_ext_arr(self, arr, as_vector)
        ti.sync()
        return arr

    @python_scope
    def from_numpy(self, arr):
        if len(arr.shape) == len(self.shape) + 1:
            as_vector = True
            assert self.m == 1, "This is not a vector field"
        else:
            as_vector = False
            assert len(arr.shape) == len(self.shape) + 2
        dim_ext = 1 if as_vector else 2
        assert len(arr.shape) == len(self.shape) + dim_ext
        from taichi.lang.meta import ext_arr_to_matrix
        ext_arr_to_matrix(arr, self, as_vector)
        ti.sync()

    @python_scope
    def __setitem__(self, key, value):
        self.initialize_host_accessors()
        if not isinstance(value, (list, tuple)):
            value = list(value)
        if not isinstance(value[0], (list, tuple)):
            value = [[i] for i in value]
        for i in range(self.n):
            for j in range(self.m):
                self[key][i, j] = value[i][j]

    @python_scope
    def __getitem__(self, key):
        self.initialize_host_accessors()
        key = self.pad_key(key)
        return ti.Matrix.with_entries(
            self.n, self.m,
            [SNodeHostAccess(e, key) for e in self.host_accessors])

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        return f'<{self.n}x{self.m} ti.Matrix.field>'


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
