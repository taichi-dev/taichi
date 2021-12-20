import math

from taichi.lang import impl, ops, matrix
from taichi.lang.kernel_impl import func, pyfunc
from taichi.types import f32, f64


@func
def _randn(dt):
    '''
    Generates a random number from standard normal distribution
    using the Box-Muller transform.
    '''
    assert dt == f32 or dt == f64
    u1 = ops.random(dt)
    u2 = ops.random(dt)
    r = ops.sqrt(-2 * ops.log(u1))
    c = ops.cos(math.tau * u2)
    return r * c


def randn(dt=None):
    """Generates a random number from standard normal distribution.

    Implementation refers to :func:`taichi.lang.random.randn`.

    Args:
        dt (DataType): The datatype for the generated random number.

    Returns:
        The generated random number.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    return _randn(dt)


@pyfunc
def _matrix_transpose(self):
    """Get the transpose of a matrix.

    Returns:
        Get the transpose of a matrix.

    """
    return matrix.Matrix([[self[i, j] for i in range(self.n)]
                   for j in range(self.m)])


@pyfunc
def _matrix_cross3d(self, other):
    return matrix.Matrix([
        self[1] * other[2] - self[2] * other[1],
        self[2] * other[0] - self[0] * other[2],
        self[0] * other[1] - self[1] * other[0],
    ])


@pyfunc
def _matrix_cross2d(self, other):
    return self[0] * other[1] - self[1] * other[0]


@pyfunc
def _matrix_outer_product(self, other):
    """Perform the outer product with the input Vector (1-D Matrix).

    Args:
        other (:class:`~taichi.lang.matrix.Matrix`): The input Vector (1-D Matrix) to perform the outer product.

    Returns:
        :class:`~taichi.lang.matrix.Matrix`: The outer product result (Matrix) of the two Vectors.

    """
    impl.static(
        impl.static_assert(self.m == 1,
                           "lhs for outer_product is not a vector"))
    impl.static(
        impl.static_assert(other.m == 1,
                           "rhs for outer_product is not a vector"))
    return matrix.Matrix([[self[i] * other[j] for j in range(other.n)]
                   for i in range(self.n)])


__all__ = ['randn']
