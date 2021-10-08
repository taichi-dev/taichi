class ArgAnyArray:
    """Type annotation for arbitrary arrays, including external arrays and Taichi ndarrays.

    For external arrays, we can treat it as a Taichi field with Vector or Matrix elements by specifying element dim and layout.
    For Taichi vector/matrix ndarrays, we will automatically identify element dim and layout. If they are explicitly specified, we will check compatibility between the actual arguments and the annotation.

    Args:
        element_dim (Union[Int, NoneType], optional): None if not specified (will be treated as 0 for external arrays), 0 if scalar elements, 1 if vector elements, and 2 if matrix elements.
        layout (Union[Layout, NoneType], optional): None if not specified (will be treated as Layout.AOS for external arrays), Layout.AOS or Layout.SOA.
    """
    def __init__(self, element_dim=None, layout=None):
        if element_dim is not None and (element_dim < 0 or element_dim > 2):
            raise ValueError(
                "Only scalars, vectors, and matrices are allowed as elements of ti.any_arr()"
            )
        self.element_dim = element_dim
        self.layout = layout

    def check_element_dim(self, arg, arg_dim):
        if self.element_dim is not None and self.element_dim != arg_dim:
            raise ValueError(
                f"Invalid argument into ti.any_arr() - required element_dim={self.element_dim}, but {arg} is provided"
            )

    def check_layout(self, arg):
        if self.layout is not None and self.layout != arg.layout:
            raise ValueError(
                f"Invalid argument into ti.any_arr() - required layout={self.layout}, but {arg} is provided"
            )


def ext_arr():
    """Type annotation for external arrays.

    External arrays are formally defined as the data from other Python frameworks.
    For now, Taichi supports numpy and pytorch.

    Example::

        >>> @ti.kernel
        >>> def to_numpy(arr: ti.ext_arr()):
        >>>     for i in x:
        >>>         arr[i] = x[i]
        >>>
        >>> arr = numpy.zeros(...)
        >>> to_numpy(arr)  # `arr` will be filled with `x`'s data.
    """
    return ArgAnyArray()


any_arr = ArgAnyArray
"""Alias for :class:`~taichi.type.annotations.ArgAnyArray`.

Example::

    >>> @ti.kernel
    >>> def to_numpy(x: ti.any_arr(), y: ti.any_arr()):
    >>>     for i in range(n):
    >>>         x[i] = y[i]
    >>>
    >>> y = ti.ndarray(ti.f64, shape=n)
    >>> ... # calculate y
    >>> x = numpy.zeros(n)
    >>> to_numpy(x, y)  # `x` will be filled with `y`'s data.
"""


class Template:
    """Type annotation for template kernel parameter.

    See also https://docs.taichi.graphics/lang/articles/advanced/meta.

    Args:
        tensor (Any): unused
        dim (Any): unused
    """
    def __init__(self, tensor=None, dim=None):
        self.tensor = tensor
        self.dim = dim


template = Template
"""Alias for :class:`~taichi.type.annotations.Template`.
"""

__all__ = ['ext_arr', 'any_arr', 'template']
