import numpy as np
from taichi.core.util import ti_core as _ti_core
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

    @python_scope
    def fill(self, val):
        """Fills ndarray with a specific scalar value.

        Args:
            val (Union[int, float]): Value to fill.
        """
        if impl.current_cfg().ndarray_use_torch:
            self.arr.fill_(val)
        else:
            from taichi.lang.meta import fill_ndarray  # pylint: disable=C0415
            fill_ndarray(self, val)

    @python_scope
    def to_numpy(self):
        """Converts ndarray to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        if impl.current_cfg().ndarray_use_torch:
            return self.arr.cpu().numpy()
        else:
            import numpy as np  # pylint: disable=C0415
            arr = np.zeros(shape=self.arr.shape,
                           dtype=to_numpy_type(self.dtype))
            from taichi.lang.meta import \
                ndarray_to_ext_arr  # pylint: disable=C0415
            ndarray_to_ext_arr(self, arr)
            impl.get_runtime().sync()
            return arr

    @python_scope
    def from_numpy(self, arr):
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
            self.arr = torch.from_numpy(arr).to(self.arr.dtype)
        else:
            if hasattr(arr, 'contiguous'):
                arr = arr.contiguous()
            from taichi.lang.meta import \
                ext_arr_to_ndarray  # pylint: disable=C0415
            ext_arr_to_ndarray(arr, self)
            impl.get_runtime().sync()

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
    """Taichi ndarray with scalar elements implemented with a torch tensor.

    Args:
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the ndarray.
    """
    def __init__(self, dtype, shape):
        super().__init__(dtype, shape)

    @property
    def shape(self):
        return tuple(self.arr.shape)

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
        else:
            self.initialize_host_accessor()
            return self.host_accessor.getter(*self.pad_key(key))

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

    def getter(self):
        if impl.current_cfg().ndarray_use_torch:
            return self.arr[self.indices]
        else:
            self.ndarr.initialize_host_accessor()
            return self.ndarr.host_accessor.getter(*self.ndarr.pad_key(self.indices))

    def setter(self, value):
        if impl.current_cfg().ndarray_use_torch:
            self.arr[self.indices] = value
        else:
            self.ndarr.initialize_host_accessor()
            self.ndarr.host_accessor.setter(value, *self.ndarr.pad_key(self.indices))
