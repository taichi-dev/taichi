from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.util import has_pytorch, python_scope, to_pytorch_type, to_taichi_type
import numbers


class Ndarray:
    """Taichi ndarray class implemented with a torch tensor.

    Args:
        dtype (DataType): Data type of the ndarray.
        shape (Union[int, tuple[int]]): Shape of the torch tensor.
    """
    def __init__(self, dtype, shape):
        if isinstance(shape, numbers.Number):
            shape = (shape, )
        assert has_pytorch(
        ), "PyTorch must be available if you want to create a Taichi ndarray."
        import torch
        if impl.current_cfg().arch == _ti_core.Arch.cuda:
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.arr = torch.empty(shape, dtype=to_pytorch_type(dtype), device=device)

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


class ScalarNdarray(Ndarray):
    """Taichi ndarray with scalar elements implemented with a torch tensor.

    Args:
        dtype (DataType): Data type of the ndarray.
        shape (Union[int, tuple[int]]): Shape of the ndarray.
    """
    def __init__(self, dtype, shape):
        super().__init__(dtype, shape)

    @property
    def shape(self):
        return tuple(self.arr.shape)

    @python_scope
    def __getitem__(self, item):
        return self.arr.__getitem__(item)

    @python_scope
    def __setitem__(self, key, value):
        return self.arr.__setitem__(key, value)

    def __repr__(self):
        return '<ti.ndarray>'
