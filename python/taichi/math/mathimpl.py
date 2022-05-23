# pylint: disable=W0622
"""
Math functions for glsl-like functions and other stuff.
"""
from math import e, pi

from taichi.lang import impl
from taichi.lang.ops import (acos, asin, atan2, ceil, cos, exp, floor, log,
                             max, min, pow, round, sin, sqrt, tan, tanh)

import taichi as ti

_get_uint_ip = lambda: ti.u32 if impl.get_runtime(
).default_ip == ti.i32 else ti.u64


def vec2(*args):
    """2D floating vector type.
    """
    return ti.types.vector(2, impl.get_runtime().default_fp)(*args)  # pylint: disable=E1101


def vec3(*args):
    """3D floating vector type.
    """
    return ti.types.vector(3, impl.get_runtime().default_fp)(*args)  # pylint: disable=E1101


def vec4(*args):
    """4D floating vector type.
    """
    return ti.types.vector(4, impl.get_runtime().default_fp)(*args)  # pylint: disable=E1101


def ivec2(*args):
    """2D signed int vector type.
    """
    return ti.types.vector(2, impl.get_runtime().default_ip)(*args)  # pylint: disable=E1101


def ivec3(*args):
    """3D signed int vector type.
    """
    return ti.types.vector(3, impl.get_runtime().default_ip)(*args)  # pylint: disable=E1101


def ivec4(*args):
    """4D signed int vector type.
    """
    return ti.types.vector(4, impl.get_runtime().default_ip)(*args)  # pylint: disable=E1101


def uvec2(*args):
    """2D unsigned int vector type.
    """
    return ti.types.vector(2, _get_uint_ip())(*args)  # pylint: disable=E1101


def uvec3(*args):
    """3D unsigned int vector type.
    """
    return ti.types.vector(3, _get_uint_ip())(*args)  # pylint: disable=E1101


def uvec4(*args):
    """4D unsigned int vector type.
    """
    return ti.types.vector(4, _get_uint_ip())(*args)  # pylint: disable=E1101


mat2 = ti.types.matrix(2, 2, impl.get_runtime().default_fp)  # pylint: disable=E1101
"""2x2 floating matrix type.
"""

mat3 = ti.types.matrix(3, 3, impl.get_runtime().default_fp)  # pylint: disable=E1101
"""3x3 floating matrix type.
"""

mat4 = ti.types.matrix(4, 4, impl.get_runtime().default_fp)  # pylint: disable=E1101
"""4x4 floating matrix type.
"""


@ti.func
def mix(x, y, a):
    """Performs a linear interpolation between `x` and `y` using
    `a` to weight between them. The return value is computed as
    :math:`x\times a + (1-a)\times y`.

    The arguments can be scalars or :class:`~taichi.Matrix`,
    as long as the operation can be performed.

    This function is similar to the `mix` function in GLSL.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specify
            the start of the range in which to interpolate.
        y (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specify
            the end of the range in which to interpolate.
        a (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specify
            the weight to use to interpolate between x and y.

    Returns:
        (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): The linear
            interpolation of `x` and `y` by weight `a`.

    Example::

        >>> x = ti.Vector([1, 1, 1])
        >>> y = ti.Vector([2, 2, 2])
        >>> a = ti.Vector([1, 0, 0])
        >>> ti.mix(x, y, a)
        [2, 1, ]
        >>> x = ti.Matrix([[1, 2], [2, 3]], ti.f32)
        >>> y = ti.Matrix([[3, 5], [4, 5]], ti.f32)
        >>> a = 0.5
        >>> ti.mix(x, y, a)
        [[2.0, 3.5], [3.0, 4.0]]
    """
    return x * (1.0 - a) + y * a


@ti.func
def clamp(x, xmin, xmax):
    """Constrain a value to lie between two further values, element-wise.
    The returned value is computed as `min(max(x, xmin), xmax)`.

    The arguments can be scalars or :class:`~taichi.Matrix`,
    as long as they can be broadcasted to a common shape.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specify
            the value to constrain.
        y (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specify
            the lower end of the range into which to constrain `x`.
        a (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specify
            the upper end of the range into which to constrain `x`.

    Returns:
        The value of `x` constrained to the range `xmin` to `xmax`.

    Example::

        >>> v = ti.Vector([0, 0.5, 1.0, 1.5])
        >>> ti.clamp(v, 0.5, 1.0)
        [0.5, 0.5, 1.0, 1.0]
        >>> x = ti.Matrix([[0, 1], [-2, 2]], ti.f32)
        >>> y = ti.Matrix([[1, 2], [1, 2]], ti.f32)
        >>> ti.clamp(x, 0.5, y)
        [[0.5, 1.0], [0.5, 2.0]]
    """
    return min(xmax, max(xmin, x))


@ti.func
def step(edge, x):
    """Generate a step function by comparing two values, element-wise.

    `step` generates a step function by comparing `x` to edge.
    For element i of the return value, 0.0 is returned if x[i] < edge[i],
    and 1.0 is returned otherwise.

    The two arguments can be scalars or :class:`~taichi.Matrix`,
    as long as they can be broadcasted to a common shape.

    Args:
        edge (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specify
            the location of the edge of the step function.
        x (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specify
            the value to be used to generate the step function.

    Returns:
        The return value is computed as `x >= edge`, with type promoted.

    Example::

        >>> x = ti.Matrix([[0, 1], [2, 3]], ti.f32)
        >>> y = 1
        >>> ti.step(x, y)
        [[1.0, 1.0], [0.0, 0.0]]
    """
    return ti.cast(x >= edge, float)


@ti.func
def fract(x):
    """Compute the fractional part of the argument, element-wise.
    It's equivalent to `x - ti.floor(x)`.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): The
            input value.

    Returns:
        The fractional part of `x`.

    Example::

        >>> x = ti.Vector([-1.2, -0.7, 0.3, 1.2])
        >>> ti.fract(x)
        [0.8, 0.3, 0.3, 0.2]
    """
    return x - ti.floor(x)


@ti.func
def smoothstep(edge0, edge1, x):
    """Performs smooth Hermite interpolation between 0 and 1 when
    `edge0 < x < edge1`, element-wise.

    The arguments can be scalars or :class:`~taichi.Matrix`,
    as long as they can be broadcasted to a common shape.

    This function is equivalent to the `smoothstep` in GLSL.

    Args:
        edge0 (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specifies
            the value of the lower edge of the Hermite function.
        edge1 (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specifies
            the value of the upper edge of the Hermite function.
        x (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): Specifies
            the source value for interpolation.

    Returns:
        The smoothly interpolated value.

    Example::

        >>> edge0 = ti.Vector([0, 1, 2])
        >>> edge1 = 1
        >>> x = ti.Vector([0.5, 1.5, 2.5])
        >>> ti.smoothstep(edge0, edge1, x)
        [0.5, 1.0, 0.0]
    """
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def sign(x):
    """Extract the sign of the parameter, element-wise.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): The
            input value.

    Returns:
        -1.0 if `x` is less than 0.0, 0.0 if `x` is equal to 0.0,
        and +1.0 if `x` is greater than 0.0.

    Example::

        >>> x = ti.Vector([-1.0, 0.0, 1.0])
        >>> ti.sign(x)
        [0.8, 0.3, 0.3, 0.2]
    """
    return ti.cast((x >= 0.0) - (x <= 0.0), float)


@ti.func
def normalize(v):
    """Calculates the unit vector in the same direction as the
    original vector `v`.

    It's equivalent to the `normalize` function is GLSL.

    Args:
        x (:class:`~taichi.Matrix`): The vector to normalize.

    Returns:
        The normalized vector :math:`v/|v|`.

    Example::

        >>> v = ti.Vector([1, 2, 3])
        >>> ti.normalize(v)
        [0.333333, 0.666667, 1.000000]
    """
    return v / v.norm()


@ti.func
def log2(x):
    """Return the base 2 logarithm of `x`, so that if :math:`2^y=x`,
    then :math:`y=\\log2(x)`.

    This is equivalent to the `log2` function is GLSL.

    Args:
        x (:class:`~taichi.Matrix`): The input value.

    Returns:
        The base 2 logarithm of `x`.

    Example::

        >>> v = ti.Vector([1., 2., 3.])
        >>> ti.log2(x)
        [0.000000, 1.000000, 1.584962]
    """
    return ti.log(x) / ti.static(ti.log(2.0))


@ti.func
def reflect(x, n):
    """Calculate the reflection direction for an incident vector.

    For a given incident vector `x` and surface normal `n` this
    function returns the reflection direction calculated as
    :math:`x - 2.0 * dot(x, n) * n`.

    This is equivalent to the `reflect` function is GLSL.

    `n` should be normalized in order to achieve the desired result.

    Args:
        x (:class:`~taichi.Matrix`): The incident vector.
        n (:class:`~taichi.Matrix`): The normal vector.

    Returns:
        The reflected vector.

    Example::

        >>> x = ti.Vector([1., 2., 3.])
        >>> n = ti.Vector([0., 1., 0.])
        >>> reflect(x, n)
        [1.0, -2.0, 3.0]
    """
    k = x.dot(n)
    return x - 2.0 * k * n


@ti.func
def degrees(x):
    """Convert `x` in radians to degrees, element-wise.

    Args:
        x (:class:`~taichi.Matrix`): The input angle in radians.

    Returns:
        angle in degrees.

    Example::

        >>> x = ti.Vector([-pi/2, pi/2])
        >>> degrees(x)
        [-90., 90.]
    """
    return x * ti.static(180.0 / pi)


@ti.func
def radians(x):
    """Convert `x` in degrees to radians, element-wise.

    Args:
        x (:class:`~taichi.Matrix`): The input angle in degrees.

    Returns:
        angle in radians.

    Example::

        >>> x = ti.Vector([-90., 45., 90.])
        >>> radians(x) / pi
        [-0.5, 0.25, 0.5]
    """
    return x * ti.static(pi / 180.0)


@ti.func
def distance(x, y):
    """Calculate the distance between two points.

    This function is equivalent to the `distance` function is GLSL.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): The first input point.
        y (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): The second input point.

    Returns:
        The distance between the two points.

    Example::

        >>> x = ti.Vector([0, 0, 0])
        >>> y = ti.Vector([1, 1, 1])
        >>> distance(x, y)
        1.732051
    """
    return (x - y).norm()


@ti.func
def refract(x, n, eta):
    """Calculate the refraction direction for an incident vector.

    This function is equivalent to the `refract` function in GLSL.

    Args:
        x (:class:`~taichi.Matrix`): The incident vector.
        n (:class:`~taichi.Matrix`): The normal vector.
        eta (float): The ratio of indices of refraction.

    Returns:
        :class:`~taichi.Matrix`: The refraction direction vector.

    Example::

        >>> x = ti.Vector([1., 1., 1.])
        >>> n = ti.Vector([0, 1., 0])
        >>> refract(x, y, 2.0)
        [2., -1., 2]
    """
    dxn = x.dot(n)
    result = ti.zero(x)
    k = 1.0 - eta * eta * (1.0 - dxn * dxn)
    if k >= 0.0:
        result = eta * x - (eta * dxn + ti.sqrt(k)) * n
    return result


@ti.func
def dot(x, y):
    """Calculate the dot product of two vectors.

    Args:
        x (:class:`~taichi.Matrix`): The first input vector.
        y (:class:`~taichi.Matrix`): The second input vector.

    Returns:
        The dot product of two vectors.

    Example::

        >>> x = ti.Vector([1., 1., 0.])
        >>> y = ti.Vector([0., 1., 1.])
        >>> dot(x, y)
        1.
    """
    return x.dot(y)


@ti.func
def cross(x, y):
    """Calculate the cross product of two vectors.

    The two input vectors must have the same dimension :math:`d <= 3`.

    This function calls the `cross` method of :class:`~taichi.Vector`.

    Args:
        x (:class:`~taichi.Matrix`): The first input vector.
        y (:class:`~taichi.Matrix`): The second input vector.

    Returns:
        The cross product of two vectors.

    Example::

        >>> x = ti.Vector([1., 0., 0.])
        >>> y = ti.Vector([0., 1., 0.])
        >>> cross(x, y)
        [0., 0., 1.]
    """
    return x.cross(y)


@ti.func
def mod(x, y):
    """Compute value of one parameter modulo another, element-wise.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): The first input.
        y (:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`): The second input.

    Returns:
        the value of `x` modulo `y`. This is computed as `x - y * floor(x/y)`.

    Example::

        >>> x = ti.Vector([-0.5, 0.5, 1.])
        >>> y = 1.0
        >>> mod(x, y)
        [0.5, 0.5, 0.0]
    """
    return x - y * ti.floor(x / y)


@ti.func
def rotate2d(p, ang):
    """Rotates a 2d vector by a given angle in counter-clockwise.

    Args:
        p (:class:`~taichi.math.vec2`): The 2d vector to rotate.
        ang (float): Angle of rotation, in radians.

    Returns:
        :class:`~taichi.math.vec2`: The vector after rotation.

    Example::

        >>> from taichi.math import *
        >>> @ti.kernel
        >>> def test():
        >>>     v = vec2(1, 0)
        >>>     print(rotate2d(v, radians(30)))
        [0.866025, 0.500000]
    """
    ca, sa = ti.cos(ang), ti.sin(ang)
    x, y = p
    return vec2(x * ca - p.y * sa, x * sa + y * ca)


@ti.func
def rotate3d(p, axis, ang):
    """Rotates a vector in 3d space, given an axis and angle of rotation.

    The vector `axis` should be a unit vector.

    See "https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula"

    Args:
        p (:class:`~taichi.math.vec3`): The 3d vector to rotate.
        axis (:class:`~taichi.math.vec3`): Axis of rotation.
        ang (float): Angle of rotation, in radians.

    Example::

        >>> from taichi.math import *
        >>> @ti.kernel
        >>> def test():
        >>>     v = vec3(1, 0, 0)
        >>>     axis = normalize(vec3(1, 1, 1))
        >>>     print(rotate3d(v, axis, radians(30)))
        [0.910684, 0.333333, -0.244017]

    Returns:
        :class:`~taichi.math.vec3`: The vector after rotation.
    """
    ca, sa = ti.cos(ang), ti.sin(ang)
    return mix(dot(p, axis) * axis, p, ca) + cross(axis, p) * sa


@ti.func
def eye(n: ti.template()):
    """Returns the nxn identiy matrix.

    Alias for :func:`~taichi.Matrix.identity`.
    """
    return ti.Matrix.identity(float, n)


@ti.func
def rot2(ang):
    """Returns the matrix representation of a 2d counter-clockwise rotation,
    given the angle of rotation.

    Args:
        ang (float): Angle of rotation in radians.

    Returns:
        :class:`~taichi.math.mat2`: 2x2 rotation matrix.

    Example::

        >>> from taichi.math import *
        >>> @ti.kernel
        >>> def test():
        >>>     M = rot2(radians(30))
        [[0.866025, -0.500000], [0.500000, 0.866025]]
    """
    ca, sa = ti.cos(ang), ti.sin(ang)
    return mat2([[ca, -sa], [sa, ca]])


@ti.func
def rot3(axis, ang):
    """Returns the matrix representation of a 3d rotation,
    given the axis and angle of rotation.

    Args:
        axis (:class:`~taichi.math.vec3`): Axis of rotation.
        ang (float): Angle of rotation in radians.

    Returns:
        :class:`~taichi.math.mat3`: 3x3 rotation matrix.

    Example::

        >>> from taichi.math import *
        >>> @ti.kernel
        >>> def test():
        >>>     M = rot3(normalize(vec3(1, 1, 1)), radians(30))
        [[0.732051, -0.366025, 0.633975],
         [0.633975, 0.732051, -0.366025],
         [-0.366025, 0.633975, 0.732051]]
    """
    ca, sa = ti.cos(ang), ti.sin(ang)
    x, y, z = axis
    I = eye(3)
    K = mat3([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return I + sa * K + (1.0 - ca) * K @ K


@ti.func
def length(x):
    """Calculate the length of a vector.

    This function is equivalent to the `length` function in GLSL.
    Args:
        x (:class:`~taichi.Matrix`): The vector of which to calculate the length.

    Returns:
        The Euclidean norm of the vector.

    Example::

        >>> x = ti.Vector([1, 1, 1])
        >>> length(x)
        1.732051
    """
    return x.norm()


@ti.func
def determinant(m):
    """Alias for :func:`taichi.Matrix.determinant`.
    """
    return m.determinant()


@ti.func
def _inverse2x2(m):
    return mat2([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]
                 ]) / (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])


@ti.func
def _inverse3x3(m):
    a00 = m[0, 0]
    a01 = m[0, 1]
    a02 = m[0, 2]
    a10 = m[1, 0]
    a11 = m[1, 1]
    a12 = m[1, 2]
    a20 = m[2, 0]
    a21 = m[2, 1]
    a22 = m[2, 2]
    b01 = a22 * a11 - a12 * a21
    b11 = -a22 * a10 + a12 * a20
    b21 = a21 * a10 - a11 * a20
    det = a00 * b01 + a01 * b11 + a02 * b21
    return mat3([[b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11)],
                 [b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10)],
                 [b21, (-a21 * a00 + a01 * a20),
                  (a11 * a00 - a01 * a10)]]) / det


@ti.func
def _inverse4x4(m):
    a00 = m[0, 0]
    a01 = m[0, 1]
    a02 = m[0, 2]
    a03 = m[0, 3]
    a10 = m[1, 0]
    a11 = m[1, 1]
    a12 = m[1, 2]
    a13 = m[1, 3]
    a20 = m[2, 0]
    a21 = m[2, 1]
    a22 = m[2, 2]
    a23 = m[2, 3]
    a30 = m[3, 0]
    a31 = m[3, 1]
    a32 = m[3, 2]
    a33 = m[3, 3]
    b00 = a00 * a11 - a01 * a10
    b01 = a00 * a12 - a02 * a10
    b02 = a00 * a13 - a03 * a10
    b03 = a01 * a12 - a02 * a11
    b04 = a01 * a13 - a03 * a11
    b05 = a02 * a13 - a03 * a12
    b06 = a20 * a31 - a21 * a30
    b07 = a20 * a32 - a22 * a30
    b08 = a20 * a33 - a23 * a30
    b09 = a21 * a32 - a22 * a31
    b10 = a21 * a33 - a23 * a31
    b11 = a22 * a33 - a23 * a32
    det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
    return mat4([[
        a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09,
        a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03
    ],
                 [
                     a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 +
                     a03 * b07, a32 * b02 - a30 * b05 - a33 * b01,
                     a20 * b05 - a22 * b02 + a23 * b01
                 ],
                 [
                     a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 -
                     a03 * b06, a30 * b04 - a31 * b02 + a33 * b00,
                     a21 * b02 - a20 * b04 - a23 * b00
                 ],
                 [
                     a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 +
                     a02 * b06, a31 * b01 - a30 * b03 - a32 * b00,
                     a20 * b03 - a21 * b01 + a22 * b00
                 ]]) / det


@ti.func
def inverse(mat):  # pylint: disable=R1710
    """Calculate the inverse of a matrix.

    This function is equivalent to the `inverse` function in GLSL.

    Args:
        mat (:class:`taichi.Matrix`): The matrix of which to take the inverse.

    Returns:
        Inverse of the input matrix.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     m = mat3([(1, 1, 0), (0, 1, 1), (0, 0, 1)])
        >>>     print(inverse(m))
        >>>
        >>> test()
        [[1.000000, -1.000000, 1.000000],
         [0.000000, 1.000000, -1.000000],
         [0.000000, 0.000000, 1.000000]]
    """
    assert mat.m == mat.n and 2 <= mat.m <= 4, "A 2x2, 3x3 or 4x4 matrix is expected"
    if ti.static(mat.m == 2):
        return _inverse2x2(mat)

    if ti.static(mat.m == 3):
        return _inverse3x3(mat)

    if ti.static(mat.m == 4):
        return _inverse4x4(mat)


__all__ = [
    "acos", "asin", "atan2", "ceil", "clamp", "cos", "cross", "degrees",
    "determinant", "distance", "dot", "e", "exp", "eye", "floor", "fract",
    "inverse", "ivec2", "ivec3", "ivec4", "length", "log", "log2", "mat2",
    "mat3", "mat4", "max", "min", "mix", "mod", "normalize", "pi", "pow",
    "radians", "reflect", "refract", "rot2", "rot3", "rotate2d", "rotate3d",
    "round", "sign", "sin", "smoothstep", "sqrt", "step", "tan", "tanh",
    "uvec2", "uvec3", "uvec4", "vec2", "vec3", "vec4"
]
