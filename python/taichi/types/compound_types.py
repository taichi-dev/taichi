import taichi


class CompoundType:
    pass


class TensorType(CompoundType):
    def __init__(self, shape, dtype):
        self.dtype = dtype
        self.shape = shape


# TODO: maybe move MatrixType, StructType here to avoid the circular import?
def matrix(n, m, dtype):
    """Creates a matrix type with given shape and data type.

    Args:
        n (int): number of rows of the matrix.
        m (int): number of columns of the matrix.
        dtype (:mod:`~taichi.types.primitive_types`): matrix data type.

    Returns:
        A matrix type.

    Example::

        >>> mat2x2 = ti.types.matrix(2, 2, ti.f32)  # 2x2 matrix type
        >>> M = mat2x2([[1., 2.], [3., 4.]])  # an instance of this type
    """
    return taichi.lang.matrix.MatrixType(n, m, dtype)


def vector(n, dtype):
    """Creates a vector type with given shape and data type.

    Args:
        n (int): dimension of the vector.
        dtype (:mod:`~taichi.types.primitive_types`): vector data type.

    Returns:
        A vector type.

    Example::

        >>> vec3 = ti.types.vector(3, ti.f32)  # 3d vector type
        >>> v = vec3([1., 2., 3.])  # an instance of this type
    """
    return taichi.lang.matrix.MatrixType(n, 1, dtype)


def struct(**kwargs):
    """Creates a struct type with given members.

    Args:
        kwargs (dict): a dictionay contains the names and types of the
            struct members.

    Returns:
        A struct type.

    Example::

        >>> vec3 = ti.types.vector(3, ti.f32)
        >>> sphere = ti.types.strcut(center=vec3, radius=float)
        >>> s = sphere(center=vec3([0., 0., 0.]), radius=1.0)
    """
    return taichi.lang.struct.StructType(**kwargs)


__all__ = ['matrix', 'vector', 'struct']
