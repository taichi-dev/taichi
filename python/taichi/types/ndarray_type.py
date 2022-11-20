import warnings

from taichi.lang.enums import Layout
from taichi.types.compound_types import (CompoundType, TensorType, matrix,
                                         vector)


class NdarrayTypeMetadata:
    def __init__(self, element_type, shape=None):
        self.element_type = element_type
        self.shape = shape
        self.layout = Layout.AOS


# TODO(Haidong): This is a helper function that creates a MatrixType
#                with respect to element_dim and element_shape.
#                Remove this function when the two args are totally deprecated.
def _make_matrix_dtype_from_element_shape(element_dim, element_shape,
                                          primitive_dtype):
    if isinstance(primitive_dtype, CompoundType):
        raise TypeError(
            f'Cannot specifiy matrix dtype "{primitive_dtype}" and element shape or dim at the same time.'
        )

    # Scalars
    if element_dim == 0 or (element_shape is not None
                            and len(element_shape) == 0):
        return primitive_dtype

    # Cook element dim and shape into matrix type.
    mat_dtype = None
    if element_dim is not None:
        if element_dim < 0 or element_dim > 2:
            raise ValueError(
                "Only scalars, vectors, and matrices are allowed as elements of ti.types.ndarray()"
            )
        # Check dim consistency. The matrix dtype will be cooked later.
        if element_shape is not None and len(element_shape) != element_dim:
            raise ValueError(
                f"Both element_shape and element_dim are specified, but shape doesn't match specified dim: {len(element_shape)}!={element_dim}"
            )
        mat_dtype = vector(None,
                           primitive_dtype) if element_dim == 1 else matrix(
                               None, None, primitive_dtype)
    if element_shape is not None:
        if len(element_shape) > 2:
            raise ValueError(
                "Only scalars, vectors, and matrices are allowed as elements of ti.types.ndarray()"
            )
        mat_dtype = vector(
            element_shape[0],
            primitive_dtype) if len(element_shape) == 1 else matrix(
                element_shape[0], element_shape[1], primitive_dtype)
    return mat_dtype


class NdarrayType:
    """Type annotation for arbitrary arrays, including external arrays (numpy ndarrays and torch tensors) and Taichi ndarrays.

    For external arrays, we treat it as a Taichi data container with Scalar, Vector or Matrix elements.
    For Taichi vector/matrix ndarrays, we will automatically identify element dimension and their corresponding axis by the dimension of datatype, say scalars, matrices or vectors.
    For example, given type annotation `ti.types.ndarray(dtype=ti.math.vec3)`, a numpy array `np.zeros(10, 10, 3)` will be recognized as a 10x10 matrix composed of vec3 elements.

    Args:
        dtype (Union[PrimitiveType, VectorType, MatrixType, NoneType], optional): None if not speicified.
        [DERPRECATED] element_dim (Union[Int, NoneType], optional): None if not specified (will be treated as 0 for external arrays), 0 if scalar elements, 1 if vector elements, and 2 if matrix elements.
        [DERPRECATED] element_shape (Union[Tuple[Int], NoneType]): None if not specified, shapes of each element. For example, element_shape must be 1d for vector and 2d tuple for matrix. This argument is ignored for external arrays for now.
        field_dim (Union[Int, NoneType]): None if not specified, number of field dimensions. This argument is ignored for external arrays for now.
    """
    def __init__(self,
                 dtype=None,
                 element_dim=None,
                 element_shape=None,
                 field_dim=None):
        # TODO(Haidong) Deprecate element shape in 1.4.0. Use dtype to manage element-level arguments.
        if element_dim is not None or element_shape is not None:
            warnings.warn(
                "The element_dim and element_shape arguments for ndarray will be deprecated in v1.4.0, use matrix dtype instead.",
                DeprecationWarning)
            self.dtype = make_matrix_dtype_from_element_shape(
                element_dim, element_shape, dtype)
        else:
            self.dtype = dtype

        self.field_dim = field_dim
        self.layout = Layout.AOS

    def check_matched(self, ndarray_type: NdarrayTypeMetadata):
        # FIXME(Haidong) Cannot use Vector/MatrixType due to circular import
        # Use the CompuoundType instead to determine the specific typs.
        # TODO Replace CompoundType with MatrixType and VectorType
        if isinstance(self.dtype, CompoundType):
            element_dim = self.dtype.ndim
            element_shape = self.dtype.get_shape()
            if None in element_shape:
                element_shape = None
            if element_dim is not None and element_dim > 0:
                if not isinstance(ndarray_type.element_type, TensorType):
                    raise TypeError(
                        f"Expect TensorType element for Ndarray with element_dim: {element_dim} > 0"
                    )
                if element_dim != len(ndarray_type.element_type.shape()):
                    raise ValueError(
                        f"Invalid argument into ti.types.ndarray() - required element_dim={element_dim}, but {len(ndarray_type.element_type.shape())} is provided"
                    )
            if element_shape is not None and len(element_shape) > 0:
                if not isinstance(ndarray_type.element_type, TensorType):
                    raise TypeError(
                        f"Expect TensorType element for Ndarray with element_shape: {element_shape}"
                    )

                if element_shape != ndarray_type.element_type.shape():
                    raise ValueError(
                        f"Invalid argument into ti.types.ndarray() - required element_shape={element_shape}, but {ndarray_type.element_type.shape()} is provided"
                    )

        if self.field_dim is not None and \
            ndarray_type.shape is not None and \
            self.field_dim != len(ndarray_type.shape):
            raise ValueError(
                f"Invalid argument into ti.types.ndarray() - required field_dim={self.field_dim}, but {ndarray_type.element_type} is provided"
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
