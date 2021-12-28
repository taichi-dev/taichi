import numpy as np
import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.enums import Layout
from taichi.lang.util import (cook_dtype, has_pytorch, python_scope,
                              to_numpy_type, to_pytorch_type, to_taichi_type)

if has_pytorch():
    import torch


class Ndarray:
    """Taichi ndarray class implemented with a torch tensor.

    Args:
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the torch tensor.
    """
    def __init__(self, dtype, shape):
        self.host_accessor = None
        if impl.current_cfg().ndarray_use_torch:
            assert has_pytorch(
            ), "PyTorch must be available if you want to create a Taichi ndarray with PyTorch as its underlying storage."
            # pylint: disable=E1101
            self.arr = torch.zeros(shape,
                                   dtype=to_pytorch_type(cook_dtype(dtype)))
            if impl.current_cfg().arch == _ti_core.Arch.cuda:
                self.arr = self.arr.cuda()

        else:
            self.arr = _ti_core.Ndarray(impl.get_runtime().prog,
                                        cook_dtype(dtype), shape)

    @property
    def shape(self):
        """Gets ndarray shape.

        Returns:
            Tuple[Int]: Ndarray shape.
        """
        raise NotImplementedError()

    @property
    def element_shape(self):
        """Gets ndarray element shape.

        Returns:
            Tuple[Int]: Ndarray element shape.
        """
        raise NotImplementedError()

    @property
    def dtype(self):
        """Gets data type of each individual value.

        Returns:
            DataType: Data type of each individual value.
        """
        return to_taichi_type(self.arr.dtype)

    @property
    def data_handle(self):
        """Gets the pointer to underlying data.

        Returns:
            int: The pointer to underlying data.
        """
        return self.arr.data_ptr()

    @python_scope
    def __setitem__(self, key, value):
        """Sets ndarray element in Python scope.

        Args:
            key (Union[List[int], int, None]): Coordinates of the ndarray element.
            value (element type): Value to set.
        """
        raise NotImplementedError()

    @python_scope
    def __getitem__(self, key):
        """Gets ndarray element in Python scope.

        Args:
            key (Union[List[int], int, None]): Coordinates of the ndarray element.

        Returns:
            element type: Value retrieved.
        """
        raise NotImplementedError()

    def ndarray_fill(self, val):
        """Fills ndarray with a specific scalar value.

        Args:
            val (Union[int, float]): Value to fill.
        """
        if impl.current_cfg().ndarray_use_torch:
            self.arr.fill_(val)
        else:
            taichi.lang.meta.fill_ndarray(self, val)

    def ndarray_matrix_fill(self, val):
        """Fills ndarray with a specific scalar value.

        Args:
            val (Union[int, float]): Value to fill.
        """
        if impl.current_cfg().ndarray_use_torch:
            self.arr.fill_(val)
        else:
            taichi.lang.meta.fill_ndarray_matrix(self, val)

    def ndarray_to_numpy(self):
        """Converts ndarray to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        if impl.current_cfg().ndarray_use_torch:
            return self.arr.cpu().numpy()

        arr = np.zeros(shape=self.arr.shape, dtype=to_numpy_type(self.dtype))
        taichi.lang.meta.ndarray_to_ext_arr(self, arr)
        impl.get_runtime().sync()
        return arr

    def ndarray_matrix_to_numpy(self, as_vector):
        """Converts matrix ndarray to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        if impl.current_cfg().ndarray_use_torch:
            return self.arr.cpu().numpy()

        arr = np.zeros(shape=self.arr.shape, dtype=to_numpy_type(self.dtype))
        taichi.lang.meta.ndarray_matrix_to_ext_arr(self, arr, as_vector)
        impl.get_runtime().sync()
        return arr

    def ndarray_from_numpy(self, arr):
        """Loads all values from a numpy array.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(self.arr.shape) != tuple(arr.shape):
            raise ValueError(
                f"Mismatch shape: {tuple(self.arr.shape)} expected, but {tuple(arr.shape)} provided"
            )
        if impl.current_cfg().ndarray_use_torch:
            self.arr = torch.from_numpy(arr).to(self.arr.dtype)  # pylint: disable=E1101
            if impl.current_cfg().arch == _ti_core.Arch.cuda:
                self.arr = self.arr.cuda()
        else:
            if hasattr(arr, 'contiguous'):
                arr = arr.contiguous()

            taichi.lang.meta.ext_arr_to_ndarray(arr, self)
            impl.get_runtime().sync()

    def ndarray_matrix_from_numpy(self, arr, as_vector):
        """Loads all values from a numpy array.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(self.arr.shape) != tuple(arr.shape):
            raise ValueError(
                f"Mismatch shape: {tuple(self.arr.shape)} expected, but {tuple(arr.shape)} provided"
            )
        if impl.current_cfg().ndarray_use_torch:
            self.arr = torch.from_numpy(arr).to(self.arr.dtype)  # pylint: disable=E1101
            if impl.current_cfg().arch == _ti_core.Arch.cuda:
                self.arr = self.arr.cuda()
        else:
            if hasattr(arr, 'contiguous'):
                arr = arr.contiguous()

            taichi.lang.meta.ext_arr_to_ndarray_matrix(arr, self, as_vector)
            impl.get_runtime().sync()

    @python_scope
    def get_element_size(self):
        """Returns the size of one element in bytes.

        Returns:
            Size in bytes.
        """
        return self.arr.element_size()

    @python_scope
    def get_nelement(self):
        """Returns the total number of elements.

        Returns:
            Total number of elements.
        """
        return self.arr.nelement()

    @python_scope
    def copy_from(self, other):
        """Copies all elements from another ndarray.

        The shape of the other ndarray needs to be the same as `self`.

        Args:
            other (Ndarray): The source ndarray.
        """
        assert isinstance(other, Ndarray)
        assert tuple(self.arr.shape) == tuple(other.arr.shape)
        taichi.lang.meta.ndarray_to_ndarray(self, other)
        impl.get_runtime().sync()

    def __deepcopy__(self, memo=None):
        """Copies all elements to a new ndarray.

        Returns:
            Ndarray: The result ndarray.
        """
        raise NotImplementedError()

    def pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key, )
        assert len(key) == len(self.arr.shape)
        return key

    def initialize_host_accessor(self):
        if self.host_accessor:
            return
        impl.get_runtime().materialize()
        self.host_accessor = NdarrayHostAccessor(self.arr)


class ScalarNdarray(Ndarray):
    """Taichi ndarray with scalar elements.

    Args:
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the ndarray.
    """
    @property
    def shape(self):
        return tuple(self.arr.shape)

    @property
    def element_shape(self):
        return ()

    @python_scope
    def __setitem__(self, key, value):
        if impl.current_cfg().ndarray_use_torch:
            self.arr.__setitem__(key, value)
        else:
            self.initialize_host_accessor()
            self.host_accessor.setter(value, *self.pad_key(key))

    @python_scope
    def __getitem__(self, key):
        if impl.current_cfg().ndarray_use_torch:
            return self.arr.__getitem__(key)
        self.initialize_host_accessor()
        return self.host_accessor.getter(*self.pad_key(key))

    @python_scope
    def fill(self, val):
        self.ndarray_fill(val)

    @python_scope
    def to_numpy(self):
        return self.ndarray_to_numpy()

    @python_scope
    def from_numpy(self, arr):
        self.ndarray_from_numpy(arr)

    def __deepcopy__(self, memo=None):
        ret_arr = ScalarNdarray(self.dtype, self.shape)
        ret_arr.copy_from(self)
        return ret_arr

    def __repr__(self):
        return '<ti.ndarray>'


class NdarrayHostAccessor:
    def __init__(self, ndarray):
        if _ti_core.is_real(ndarray.dtype):

            def getter(*key):
                return ndarray.read_float(key)

            def setter(value, *key):
                ndarray.write_float(key, value)
        else:
            if _ti_core.is_signed(ndarray.dtype):

                def getter(*key):
                    return ndarray.read_int(key)
            else:

                def getter(*key):
                    return ndarray.read_uint(key)

            def setter(value, *key):
                ndarray.write_int(key, value)

        self.getter = getter
        self.setter = setter


class NdarrayHostAccess:
    """Class for accessing VectorNdarray/MatrixNdarray in Python scope.
    Args:
        arr (Union[VectorNdarray, MatrixNdarray]): See above.
        indices_first (Tuple[Int]): Indices of first-level access (coordinates in the field).
        indices_second (Tuple[Int]): Indices of second-level access (indices in the vector/matrix).
    """
    def __init__(self, arr, indices_first, indices_second):
        self.ndarr = arr
        self.arr = arr.arr
        if arr.layout == Layout.SOA:
            self.indices = indices_second + indices_first
        else:
            self.indices = indices_first + indices_second

        if impl.current_cfg().ndarray_use_torch:

            def getter():
                return self.arr[self.indices]

            def setter(value):
                self.arr[self.indices] = value
        else:

            def getter():
                self.ndarr.initialize_host_accessor()
                return self.ndarr.host_accessor.getter(
                    *self.ndarr.pad_key(self.indices))

            def setter(value):
                self.ndarr.initialize_host_accessor()
                self.ndarr.host_accessor.setter(
                    value, *self.ndarr.pad_key(self.indices))

        self.getter = getter
        self.setter = setter
