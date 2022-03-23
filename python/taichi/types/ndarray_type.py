class NdarrayType:
    """Type annotation for arbitrary arrays, including external arrays (numpy ndarrays and torch tensors) and Taichi ndarrays.

    For external arrays, we can treat it as a Taichi field with Vector or Matrix elements by specifying element dim and layout.
    For Taichi vector/matrix ndarrays, we will automatically identify element dim and layout. If they are explicitly specified, we will check compatibility between the actual arguments and the annotation.

    Args:
        element_dim (Union[Int, NoneType], optional): None if not specified (will be treated as 0 for external arrays), 0 if scalar elements, 1 if vector elements, and 2 if matrix elements.
        element_shape (Union[Tuple[Int], NoneType]): None if not specified, shapes of each element. For example, element_shape must be 1d for vector and 2d tuple for matrix. This argument is ignored for external arrays for now.
        field_dim (Union[Int, NoneType]): None if not specified, number of field dimensions. This argument is ignored for external arrays for now.
        layout (Union[Layout, NoneType], optional): None if not specified (will be treated as Layout.AOS for external arrays), Layout.AOS or Layout.SOA.
    """
    def __init__(self,
                 element_dim=None,
                 element_shape=None,
                 field_dim=None,
                 layout=None):
        if element_dim is not None and (element_dim < 0 or element_dim > 2):
            raise ValueError(
                "Only scalars, vectors, and matrices are allowed as elements of ti.types.ndarray()"
            )
        if element_dim is not None and element_shape is not None and len(
                element_shape) != element_dim:
            raise ValueError(
                f"Both element_shape and element_dim are specified, but shape doesn't match specified dim: {len(element_shape)}!={element_dim}"
            )
        self.element_shape = element_shape
        self.element_dim = len(
            element_shape) if element_shape is not None else element_dim
        self.field_dim = field_dim
        self.layout = layout

    def _check_element_dim(self, arg, arg_dim):
        if self.element_dim is not None and self.element_dim != arg_dim:
            raise ValueError(
                f"Invalid argument into ti.types.ndarray() - required element_dim={self.element_dim}, but {arg} is provided"
            )

    def _check_layout(self, arg):
        if self.layout is not None and self.layout != arg.layout:
            raise ValueError(
                f"Invalid argument into ti.types.ndarray() - required layout={self.layout}, but {arg} is provided"
            )

    def _check_element_shape(self, shapes):
        if self.element_shape is not None and shapes != self.element_shape:
            raise ValueError(
                f"Invalid argument into ti.types.ndarray() - required element_shape={self.element_shape}, but {shapes} is provided"
            )

    def _check_field_dim(self, field_dim):
        if self.field_dim is not None and field_dim != self.field_dim:
            raise ValueError(
                f"Invalid argument into ti.types.ndarray() - required field_dim={self.field_dim}, but {field_dim} is provided"
            )


ndarray = NdarrayType
"""Alias for :class:`~taichi.types.ndarray_type.NdarrayType`.

Example::

    >>> @ti.kernel
    >>> def to_numpy(x: ti.types.ndarray(), y: ti.types.ndarray()):
    >>>     for i in range(n):
    >>>         x[i] = y[i]
    >>>
    >>> y = ti.ndarray(ti.f64, shape=n)
    >>> ... # calculate y
    >>> x = numpy.zeros(n)
    >>> to_numpy(x, y)  # `x` will be filled with `y`'s data.
"""

__all__ = ['ndarray']
