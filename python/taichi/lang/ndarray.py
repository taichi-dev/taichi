from taichi.lang.util import has_pytorch, python_scope, to_taichi_type


class Ndarray:
    """Taichi ndarray class implemented with an external array.

    Currently, a torch tensor or a numpy ndarray is supported.

    Args:
        arr: The external array.
    """
    def __init__(self, arr):
        import numpy as np
        valid = isinstance(arr, np.ndarray)
        if not valid and has_pytorch():
            import torch
            valid = isinstance(arr, torch.Tensor)
        assert valid, "Only a torch tensor or a numpy ndarray is supported as Taichi ndarray implementation now."
        self.arr = arr

    @property
    def shape(self):
        """Gets ndarray shape.

        Returns:
            Tuple[Int]: Ndarray shape.
        """
        return tuple(self.arr.shape)

    @property
    def dtype(self):
        """Gets data type of each individual value.

        Returns:
            DataType: Data type of each individual value.
        """
        return to_taichi_type(self.arr.dtype)

    @python_scope
    def __getitem__(self, item):
        return self.arr.__getitem__(item)

    @python_scope
    def __setitem__(self, key, value):
        return self.arr.__setitem__(key, value)
