from taichi.lang import ops
from taichi.lang.kernel_impl import func

from .mathimpl import dot, vec2


@func
def cmul(z1, z2):
    """Performs complex multiplication between two 2d vectors.

    This is equivalent to the multiplication in the complex number field
    when `z1` and `z2` are treated as complex numbers.

    Args:
        z1 (:class:`~taichi.math.vec2`): The first input.
        z2 (:class:`~taichi.math.vec2`): The second input.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     z1 = ti.math.vec2(1, 1)
        >>>     z2 = ti.math.vec2(0, 1)
        >>>     ti.math.cmul(z1, z2)  # [-1, 1]

    Returns:
        :class:`~taichi.math.vec2`: the complex multiplication `z1 * z2`.
    """
    x1, y1 = z1[0], z1[1]
    x2, y2 = z2[0], z2[1]
    return vec2(x1 * x2 - y1 * y2, x1 * y2 + x2 * y1)


@func
def cconj(z):
    """Returns the complex conjugate of a 2d vector.

    If `z=(x, y)` then the conjugate of `z` is `(x, -y)`.

    Args:
        z (:class:`~taichi.math.vec2`): The input.

    Returns:
       :class:`~taichi.math.vec2`: The complex conjugate of `z`.
    """
    return vec2(z[0], -z[1])


@func
def cdiv(z1, z2):
    """Performs complex division between two 2d vectors.

    This is equivalent to the division in the complex number field
    when `z1` and `z2` are treated as complex numbers.

    Args:
        z1 (:class:`~taichi.math.vec2`): The first input.
        z2 (:class:`~taichi.math.vec2`): The second input.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     z1 = ti.math.vec2(1, 1)
        >>>     z2 = ti.math.vec2(0, 1)
        >>>     ti.math.cdiv(z1, z2)  # [1, -1]

    Returns:
        :class:`~taichi.math.vec2`: the complex division of `z1 / z2`.
    """
    x1, y1 = z1[0], z1[1]
    x2, y2 = z2[0], z2[1]
    return vec2(x1 * x2 + y1 * y2, -x1 * y2 + x2 * y1) / dot(z2, z2)


@func
def csqrt(z):
    """Returns the complex square root of a 2d vector `z`, so that
    if `w^2=z`, then `w = csqrt(z)`.

    Among the two square roots of `z`, if their real parts are non-zero,
    the one with positive real part is returned. If both their real parts
    are zero, the one with non-negative imaginary part is returned.

    Args:
        z (:class:`~taichi.math.vec2`): The input.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     z = ti.math.vec2(-1, 0)
        >>>     w = ti.math.csqrt(z)  # [0, 1]

    Returns:
        :class:`~taichi.math.vec2`: The complex square root.
    """
    result = vec2(0.0)
    if any(z):
        r = ops.sqrt(z.norm())
        a = ops.atan2(z[1], z[0])
        result = r * vec2(ops.cos(a / 2.0), ops.sin(a / 2.0))

    return result


@func
def cinv(z):
    """Computes the reciprocal of a complex `z`.

    Args:
        z (:class:`~taichi.math.vec2`): The input.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     z = ti.math.vec2(1, 1)
        >>>     w = ti.math.cinv(z)  # [0.5, -0.5]

    Returns:
        :class:`~taichi.math.vec2`: The reciprocal of `z`.
    """
    return cconj(z) / dot(z, z)


@func
def cpow(z, n):
    """Computes the power of a complex `z`: :math:`z^a`.

    Args:
        z (:class:`~taichi.math.vec2`): The base.
        a (float): The exponent.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     z = ti.math.vec2(1, 1)
        >>>     w = ti.math.cpow(z)  # [-2, 2]

    Returns:
        :class:`~taichi.math.vec2`: The power :math:`z^a`.
    """
    result = vec2(0.0)
    if any(z):
        r2 = dot(z, z)
        a = ops.atan2(z[1], z[0]) * n
        result = ops.pow(r2, n / 2.0) * vec2(ops.cos(a), ops.sin(a))

    return result


@func
def cexp(z):
    """Returns the complex exponential :math:`e^z`.

    `z` is a 2d vector treated as a complex number.

    Args:
        z (:class:`~taichi.math.vec2`): The exponent.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     z = ti.math.vec2(1, 1)
        >>>     w = ti.math.cexp(z)  # [1.468694, 2.287355]

    Returns:
        :class:`~taichi.math.vec2`: The power :math:`exp(z)`
    """
    r = ops.exp(z[0])
    return vec2(r * ops.cos(z[1]), r * ops.sin(z[1]))


@func
def clog(z):
    """Returns the complex logarithm of `z`, so that if :math:`e^w = z`,
    then :math:`log(z) = w`.

    `z` is a 2d vector treated as a complex number. The argument of :math:`w`
    lies in the range (-pi, pi].

    Args:
        z (:class:`~taichi.math.vec2`): The input.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     z = ti.math.vec2(1, 1)
        >>>     w = ti.math.clog(z)  # [0.346574, 0.785398]

    Returns:
        :class:`~taichi.math.vec2`: The logarithm of `z`.
    """
    ang = ops.atan2(z[1], z[0])
    r2 = dot(z, z)
    return vec2(ops.log(r2) / 2.0, ang)


__all__ = ["cconj", "cdiv", "cexp", "cinv", "clog", "cmul", "cpow", "csqrt"]
