from functools import reduce
from operator import mul
from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.util import python_scope, to_numpy_type, to_pytorch_type
from taichi.misc.util import warning
import numbers
import taichi as ti


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
        assert len(tensor_shape) in [0, 2], "Only scalars, vectors and matrices are supported"
        self.vars = vars
        self.tshape = tensor_shape
        self.host_accessors = None
        self.grad = None

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
    def name(self):
        """Gets field name.

        Returns:
            str: Field name.
        """
        return self.snode.name

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

    def loop_range(self):
        return self.vars[0]

    @python_scope
    def set_grad(self, grad):
        self.grad = grad

    @python_scope
    def get_scalar_field(self, *indices):
        """Creates a scalar field using a field member
        Only used for quant.
        """
        assert self.is_tensor, "get_scalar_field can only be called on a Matrix field"
        assert len(indices) in [1, 2]
        i = indices[0]
        j = 0 if len(indices) == 1 else indices[1]
        return SNodeField([self.vars[i * self.m + j]], ())

    @property
    def n(self):
        assert self.is_tensor
        return self.tensor_shape[0]

    @property
    def m(self):
        assert self.is_tensor
        return self.tensor_shape[1]

    @python_scope
    def fill(self, val):
        """Fills the whole field with a specific value.

        Args:
            val (Union[int, float]): Value to fill.
        """
        # TODO: avoid too many template instantiations
        """Fill the element with values.

        Args:
            val (Union[Number, List, Tuple, Matrix]): the dimension of val should be consistent with the dimension of element.

        Examples:

            Fill a scalar field:

            >>> v = ti.field(float,10)
            >>> v.fill(10.0)

            Fill a vector field:

            >>> v = ti.Vector.field(2, float,4)
            >>> v.fill([10.0,11.0])

        """
        if self.is_tensor:
            if isinstance(val, numbers.Number):
                val = tuple([tuple([val for _ in range(self.m)]) for _ in range(self.n)])
            elif isinstance(val, (list, tuple)) and isinstance(val[0], numbers.Number):
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
        else:
            from taichi.lang.meta import fill_tensor
            fill_tensor(self, val)

    @python_scope
    def to_numpy(self, keep_dims=False, as_vector=None, dtype=None):
        """Converts the taichi field to a numpy array.

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
        import numpy as np
        if self.is_tensor:
            as_vector = self.m == 1 and not keep_dims
            shape_ext = (self.n, ) if as_vector else (self.n, self.m)
            arr = np.zeros(self.shape + shape_ext, dtype=dtype)
            from taichi.lang.meta import matrix_to_ext_arr
            matrix_to_ext_arr(self, arr, as_vector)
        else:
            from taichi.lang.meta import tensor_to_ext_arr
            arr = np.zeros(shape=self.shape, dtype=dtype)
            tensor_to_ext_arr(self, arr)
        ti.sync()
        return arr

    @python_scope
    def to_torch(self, device=None, keep_dims=False):
        """Converts the taichi field to a torch tensor.

        Args:
            device (torch.device, optional): The desired device of returned tensor.
            keep_dims (bool, optional): Whether to keep the dimension after conversion.
                See :meth:`~taichi.lang.field.Field.to_numpy` for more detailed explanation.

        Returns:
            torch.tensor: The result torch tensor.
        """
        import torch
        if self.is_tensor:
            as_vector = self.m == 1 and not keep_dims
            shape_ext = (self.n, ) if as_vector else (self.n, self.m)
            arr = torch.empty(self.shape + shape_ext,
                              dtype=to_pytorch_type(self.dtype),
                              device=device)
            from taichi.lang.meta import matrix_to_ext_arr
            matrix_to_ext_arr(self, arr, as_vector)
        else:
            arr = torch.zeros(size=self.shape,
                              dtype=to_pytorch_type(self.dtype),
                              device=device)
            from taichi.lang.meta import tensor_to_ext_arr
            tensor_to_ext_arr(self, arr)
        ti.sync()
        return arr

    @python_scope
    def from_numpy(self, arr):
        """Loads all elements from a numpy array.

        The shape of the numpy array needs to be the same as the internal data structure.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        if self.is_tensor:
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
        else:
            assert len(self.shape) == len(arr.shape)
            for i in range(len(self.shape)):
                assert self.shape[i] == arr.shape[i]
            from taichi.lang.meta import ext_arr_to_tensor
            if hasattr(arr, 'contiguous'):
                arr = arr.contiguous()
            ext_arr_to_tensor(arr, self)
        ti.sync()

    @python_scope
    def from_torch(self, arr):
        """Loads all elements from a torch tensor.

        The shape of the torch tensor needs to be the same as the internal data structure.

        Args:
            arr (torch.tensor): The source torch tensor.
        """
        self.from_numpy(arr.contiguous())

    @python_scope
    def copy_from(self, other):
        assert isinstance(other, SNodeField)
        from taichi.lang.meta import tensor_to_tensor
        assert len(self.shape) == len(other.shape)
        tensor_to_tensor(self, other)

    def __str__(self):
        if impl.inside_kernel():
            return self.__repr__()  # make pybind11 happy, see Matrix.__str__
        else:
            return str(self.to_numpy())

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        if self.is_tensor:
            return f'<{self.n}x{self.m} ti.Matrix.field>'
        else:
            return '<ti.field>'

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
        self.initialize_host_accessors()
        if self.is_tensor:
            if not isinstance(value, (list, tuple)):
                value = list(value)
            if not isinstance(value[0], (list, tuple)):
                value = [[i] for i in value]
            for i in range(self.n):
                for j in range(self.m):
                    self[key][i, j] = value[i][j]
        else:
            self.host_accessors[0].setter(value, *self.pad_key(key))

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
        self.initialize_host_accessors()
        key = self.pad_key(key)
        if self.is_tensor:
            return ti.Matrix.with_entries(*self.tensor_shape, [SNodeFieldHostAccess(e, key) for e in self.host_accessors])
        else:
            return self.host_accessors[0].getter(*key)

    @python_scope
    def pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key, )
        assert len(key) == len(self.shape)
        return key + ((0, ) * (_ti_core.get_max_num_indices() - len(key)))

    @python_scope
    def initialize_host_accessors(self):
        if self.host_accessors:
            return
        impl.get_runtime().materialize()
        self.host_accessors = [SNodeFieldHostAccessor(e.ptr.snode()) for e in self.vars]


class SNodeFieldHostAccessor:
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


class SNodeFieldHostAccess:
    def __init__(self, accessor, key):
        self.accessor = accessor
        self.key = key
