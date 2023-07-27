import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.enums import Layout
from taichi.lang.exception import TaichiIndexError
from taichi.lang.util import cook_dtype, get_traceback, python_scope, to_numpy_type
from taichi.types import primitive_types
from taichi.types.ndarray_type import NdarrayTypeMetadata
from taichi.types.utils import is_real, is_signed


class Ndarray:
    """Taichi ndarray class.

    Args:
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the Ndarray.
    """

    def __init__(self):
        self.host_accessor = None
        self.shape = None
        self.element_type = None
        self.dtype = None
        self.arr = None
        self.layout = Layout.AOS
        self.grad = None

    def get_type(self):
        return NdarrayTypeMetadata(self.element_type, self.shape, self.grad is not None)

    @property
    def element_shape(self):
        """Gets ndarray element shape.

        Returns:
            Tuple[Int]: Ndarray element shape.
        """
        raise NotImplementedError()

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
        if impl.current_cfg().arch != _ti_core.Arch.cuda and impl.current_cfg().arch != _ti_core.Arch.x64:
            self._fill_by_kernel(val)
        elif _ti_core.is_tensor(self.element_type):
            self._fill_by_kernel(val)
        elif self.dtype == primitive_types.f32:
            impl.get_runtime().prog.fill_float(self.arr, val)
        elif self.dtype == primitive_types.i32:
            impl.get_runtime().prog.fill_int(self.arr, val)
        elif self.dtype == primitive_types.u32:
            impl.get_runtime().prog.fill_uint(self.arr, val)
        else:
            self._fill_by_kernel(val)

    @python_scope
    def _ndarray_to_numpy(self):
        """Converts ndarray to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        arr = np.zeros(shape=self.arr.total_shape(), dtype=to_numpy_type(self.dtype))
        from taichi._kernels import ndarray_to_ext_arr  # pylint: disable=C0415

        ndarray_to_ext_arr(self, arr)
        impl.get_runtime().sync()
        return arr

    @python_scope
    def _ndarray_matrix_to_numpy(self, as_vector):
        """Converts matrix ndarray to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        arr = np.zeros(shape=self.arr.total_shape(), dtype=to_numpy_type(self.dtype))
        from taichi._kernels import ndarray_matrix_to_ext_arr  # pylint: disable=C0415

        layout_is_aos = 1
        ndarray_matrix_to_ext_arr(self, arr, layout_is_aos, as_vector)
        impl.get_runtime().sync()
        return arr

    @python_scope
    def _ndarray_from_numpy(self, arr):
        """Loads all values from a numpy array.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(self.arr.total_shape()) != tuple(arr.shape):
            raise ValueError(f"Mismatch shape: {tuple(self.arr.shape)} expected, but {tuple(arr.shape)} provided")
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        from taichi._kernels import ext_arr_to_ndarray  # pylint: disable=C0415

        ext_arr_to_ndarray(arr, self)
        impl.get_runtime().sync()

    @python_scope
    def _ndarray_matrix_from_numpy(self, arr, as_vector):
        """Loads all values from a numpy array.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(self.arr.total_shape()) != tuple(arr.shape):
            raise ValueError(
                f"Mismatch shape: {tuple(self.arr.total_shape())} expected, but {tuple(arr.shape)} provided"
            )
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        from taichi._kernels import ext_arr_to_ndarray_matrix  # pylint: disable=C0415

        layout_is_aos = 1
        ext_arr_to_ndarray_matrix(arr, self, layout_is_aos, as_vector)
        impl.get_runtime().sync()

    @python_scope
    def _get_element_size(self):
        """Returns the size of one element in bytes.

        Returns:
            Size in bytes.
        """
        return self.arr.element_size()

    @python_scope
    def _get_nelement(self):
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
        from taichi._kernels import ndarray_to_ndarray  # pylint: disable=C0415

        ndarray_to_ndarray(self, other)
        impl.get_runtime().sync()

    def _set_grad(self, grad):
        """Sets the gradient ndarray.

        Args:
            grad (Ndarray): The gradient ndarray.
        """
        self.grad = grad

    def __deepcopy__(self, memo=None):
        """Copies all elements to a new ndarray.

        Returns:
            Ndarray: The result ndarray.
        """
        raise NotImplementedError()

    def _fill_by_kernel(self, val):
        """Fills ndarray with a specific scalar value using a ti.kernel.

        Args:
            val (Union[int, float]): Value to fill.
        """
        raise NotImplementedError()

    @python_scope
    def _pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key,)
        if len(key) != len(self.arr.total_shape()):
            raise TaichiIndexError(f"{len(self.arr.total_shape())}d ndarray indexed with {len(key)}d indices: {key}")
        return key

    @python_scope
    def _initialize_host_accessor(self):
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

    def __init__(self, dtype, arr_shape):
        super().__init__()
        self.dtype = cook_dtype(dtype)
        self.arr = impl.get_runtime().prog.create_ndarray(
            self.dtype, arr_shape, layout=Layout.NULL, zero_fill=True, dbg_info=_ti_core.DebugInfo(get_traceback())
        )
        self.shape = tuple(self.arr.shape)
        self.element_type = dtype

    def __del__(self):
        if impl is not None and impl.get_runtime() is not None and impl.get_runtime().prog is not None:
            impl.get_runtime().prog.delete_ndarray(self.arr)

    @property
    def element_shape(self):
        return ()

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessor()
        self.host_accessor.setter(value, *self._pad_key(key))

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessor()
        return self.host_accessor.getter(*self._pad_key(key))

    @python_scope
    def to_numpy(self):
        return self._ndarray_to_numpy()

    @python_scope
    def from_numpy(self, arr):
        self._ndarray_from_numpy(arr)

    def __deepcopy__(self, memo=None):
        ret_arr = ScalarNdarray(self.dtype, self.shape)
        ret_arr.copy_from(self)
        return ret_arr

    def _fill_by_kernel(self, val):
        from taichi._kernels import fill_ndarray  # pylint: disable=C0415

        fill_ndarray(self, val)

    def __repr__(self):
        return "<ti.ndarray>"


class NdarrayHostAccessor:
    def __init__(self, ndarray):
        dtype = ndarray.element_data_type()
        if is_real(dtype):

            def getter(*key):
                return ndarray.read_float(key)

            def setter(value, *key):
                ndarray.write_float(key, value)

        else:
            if is_signed(dtype):

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
        self.indices = indices_first + indices_second

        def getter():
            self.ndarr._initialize_host_accessor()
            return self.ndarr.host_accessor.getter(*self.ndarr._pad_key(self.indices))

        def setter(value):
            self.ndarr._initialize_host_accessor()
            self.ndarr.host_accessor.setter(value, *self.ndarr._pad_key(self.indices))

        self.getter = getter
        self.setter = setter


__all__ = ["Ndarray", "ScalarNdarray"]
