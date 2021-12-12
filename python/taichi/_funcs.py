import math

from taichi.lang import impl, ops
from taichi.lang.kernel_impl import func
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


__all__ = ['randn']
