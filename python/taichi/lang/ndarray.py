from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.enums import Layout
from taichi.lang.util import (cook_dtype, has_pytorch, python_scope,
                              to_pytorch_type, to_taichi_type)


class Ndarray:
    """Taichi ndarray class implemented with a torch tensor.

    Args:
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the torch tensor.
    """
    def __init__(self, dtype, shape):
        assert has_pytorch(
        ), "PyTorch must be available if you want to create a Taichi ndarray."
        import torch
        self.arr = torch.zeros(shape, dtype=to_pytorch_type(cook_dtype(dtype)))
        if impl.current_cfg().arch == _ti_core.Arch.cuda:
            self.arr = self.arr.cuda()

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
        self.arr.fill_(val)

    @python_scope
    def to_numpy(self):
        """Converts ndarray to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        return self.arr.cpu().numpy()

    @python_scope
    def from_numpy(self, arr):
        """Loads all values from a numpy array.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        import numpy as np
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(self.arr.shape) != tuple(arr.shape):
            raise ValueError(
                f"Mismatch shape: {tuple(self.arr.shape)} expected, but {tuple(arr.shape)} provided"
            )
        import torch
        self.arr = torch.from_numpy(arr).to(self.arr.dtype)


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
        self.arr.__setitem__(key, value)

    @python_scope
    def __getitem__(self, key):
        return self.arr.__getitem__(key)

    def __repr__(self):
        return '<ti.ndarray>'


class NdarrayHostAccess:
    """Class for accessing VectorNdarray/MatrixNdarray in Python scope.
    Args:
        arr (Union[VectorNdarray, MatrixNdarray]): See above.
        indices_first (Tuple[Int]): Indices of first-level access (coordinates in the field).
        indices_second (Tuple[Int]): Indices of second-level access (indices in the vector/matrix).
    """
    def __init__(self, arr, indices_first, indices_second):
        self.arr = arr.arr
        if arr.layout == Layout.SOA:
            self.indices = indices_second + indices_first
        else:
            self.indices = indices_first + indices_second

    def getter(self):
        return self.arr[self.indices]

    def setter(self, value):
        self.arr[self.indices] = value
