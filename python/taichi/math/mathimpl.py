# pylint: disable=W0622
"""
Math functions for glsl-like functions and other stuff.
"""
from math import e, inf, nan, pi

from taichi.lang import impl
from taichi.lang.ops import (acos, asin, atan2, ceil, cos, exp, floor, log,
                             max, min, pow, round, sin, sqrt, tan, tanh, unary)

import taichi as ti

cfg = impl.default_cfg

vec2 = ti.types.vector(2, cfg().default_fp)
"""2D floating vector type.
"""

vec3 = ti.types.vector(3, cfg().default_fp)
"""3D floating vector type.
"""

vec4 = ti.types.vector(4, cfg().default_fp)
"""4D floating vector type.
"""

ivec2 = ti.types.vector(2, cfg().default_ip)
"""2D signed int vector type.
"""

ivec3 = ti.types.vector(3, cfg().default_ip)
"""3D signed int vector type.
"""

ivec4 = ti.types.vector(4, cfg().default_ip)
"""3D signed int vector type.
"""

uvec2 = ti.types.vector(2, cfg().default_up)
"""2D unsigned int vector type.
"""

uvec3 = ti.types.vector(3, cfg().default_up)
"""3D unsigned int vector type.
"""

uvec4 = ti.types.vector(4, cfg().default_up)
"""4D unsigned int vector type.
"""

mat2 = ti.types.matrix(2, 2, cfg().default_fp)
"""2x2 floating matrix type.
"""

mat3 = ti.types.matrix(3, 3, cfg().default_fp)
"""3x3 floating matrix type.
"""

mat4 = ti.types.matrix(4, 4, cfg().default_fp)
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
def translate(dx, dy, dz):
    """Constructs a translation Matrix with shape (4, 4).

    Args:
        dx (float): delta x.
        dy (float): delta y.
        dz (float): delta z.

    Returns:
        :class:`~taichi.math.mat4`: translation matrix.

    Example::

        >>> import math
        >>> ti.Matrix.translate(1, 2, 3)
        [[ 1 0 0 1]
         [ 0 1 0 2]
         [ 0 0 1 3]
         [ 0 0 0 1]]
    """
    return mat4([[1., 0., 0., dx], [0., 1., 0., dy], [0., 0., 1., dz],
                 [0., 0., 0., 1.]])


@ti.func
def scale(sx, sy, sz):
    """Constructs a scale Matrix with shape (4, 4).

    Args:
        sx (float): scale x.
        sy (float): scale y.
        sz (float): scale z.

    Returns:
        :class:`~taichi.math.mat4`: scale matrix.

    Example::

        >>> import math
        >>> ti.Matrix.scale(1, 2, 3)
        [[ 1 0 0 0]
         [ 0 2 0 0]
         [ 0 0 3 0]
         [ 0 0 0 1]]
    """
    return mat4([[sx, 0., 0., 0.], [0., sy, 0., 0.], [0., 0., sz, 0.],
                 [0., 0., 0., 1.]])


@ti.func
def rot_by_axis(axis, ang):
    """Returns the 4x4 matrix representation of a 3d rotation with given axis `axis` and angle `ang`.

    Args:
        axis (vec3): rotation axis
        ang (float): angle in radians unit

    Returns:
        :class:`~taichi.math.mat4`: rotation matrix
    """
    c = ti.cos(ang)
    s = ti.sin(ang)

    axis = normalize(axis)
    temp = (1 - c) * axis
    return mat4([[
        c + temp[0] * axis[0], temp[0] * axis[1] + s * axis[2],
        temp[0] * axis[2] - s * axis[1], 0.
    ],
                 [
                     temp[1] * axis[0] - s * axis[2], c + temp[1] * axis[1],
                     temp[1] * axis[2] + s * axis[0], 0.
                 ],
                 [
                     temp[2] * axis[0] + s * axis[1],
                     temp[2] * axis[1] - s * axis[0], c + temp[2] * axis[2], 0.
                 ], [0., 0., 0., 1.]])


@ti.func
def rot_yaw_pitch_roll(yaw, pitch, roll):
    """Returns a 4x4 homogeneous rotation matrix representing the 3d rotation with Euler angles (rotate with Y axis first, X axis second, Z axis third).

    Args:
        yaw   (float): yaw angle in radians unit
        pitch (float): pitch angle in radians unit
        roll  (float): roll angle in radians unit

    Returns:
        :class:`~taichi.math.mat4`: rotation matrix
    """
    ch = ti.cos(yaw)
    sh = ti.sin(yaw)
    cp = ti.cos(pitch)
    sp = ti.sin(pitch)
    cb = ti.cos(roll)
    sb = ti.sin(roll)

    return mat4(
        [[ch * cb + sh * sp * sb, sb * cp, -sh * cb + ch * sp * sb, 0.],
         [-ch * sb + sh * sp * cb, cb * cp, sb * sh + ch * sp * cb, 0.],
         [sh * cp, -sp, ch * cp, 0.], [0., 0., 0., 1.]])


@ti.func
def rotation2d(ang):
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
        >>>     M = rotation2d(radians(30))
        [[0.866025, -0.500000], [0.500000, 0.866025]]
    """
    ca, sa = ti.cos(ang), ti.sin(ang)
    return mat2([[ca, -sa], [sa, ca]])


@ti.func
def rotation3d(ang_x, ang_y, ang_z):
    """Returns a 4x4 homogeneous rotation matrix representing the 3d rotation with Euler angles (rotate with Y axis first, X axis second, Z axis third).

    Args:
        ang_x (float): angle in radians unit around X axis
        ang_y (float): angle in radians unit around Y axis
        ang_z (float): angle in radians unit around Z axis
    Returns:
        :class:`~taichi.math.mat4`: rotation matrix
    Example:
        >>> import math
        >>> rotation3d(0.52, -0.785, 1.046)
        [[ 0.05048351 -0.61339645 -0.78816002  0.        ]
        [ 0.65833154  0.61388511 -0.4355969   0.        ]
        [ 0.75103329 -0.49688014  0.4348093   0.        ]
        [ 0.          0.          0.          1.        ]]
    """
    return rot_yaw_pitch_roll(ang_z, ang_x, ang_y)


@ti.func
def eye(n: ti.template()):
    """Returns the nxn identity matrix.

    Alias for :func:`~taichi.Matrix.identity`.
    """
    return ti.Matrix.identity(float, n)


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
    return mat.inverse()


@unary
@ti.func
def isinf(x):
    """Determines whether the parameter is positive or negative infinity, element-wise.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`taichi.Matrix`): The input.

    Example:

       >>> @ti.kernel
       >>> def test():
       >>>     x = vec4(inf, -inf, nan, 1)
       >>>     ti.math.isinf(x)
       >>>
       >>> test()
       [1, 1, 0, 0]

    Returns:
        For each element i of the result, returns 1 if x[i] is posititve or negative floating point infinity and 0 otherwise.
    """
    ftype = impl.get_runtime().default_fp
    fx = ti.cast(x, ftype)
    if ti.static(ftype == ti.f64):
        y = ti.bit_cast(fx, ti.u64)
        return (ti.cast(y >> 32, ti.u32)
                & 0x7fffffff) == 0x7ff00000 and (ti.cast(y, ti.u32) == 0)

    y = ti.bit_cast(fx, ti.u32)
    return (y & 0x7fffffff) == 0x7f800000


@unary
@ti.func
def isnan(x):
    """Determines whether the parameter is a number, element-wise.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`taichi.Matrix`): The input.

    Example:

       >>> @ti.kernel
       >>> def test():
       >>>     x = vec4(nan, -nan, inf, 1)
       >>>     ti.math.isnan(x)
       >>>
       >>> test()
       [1, 1, 0, 0]

    Returns:
        For each element i of the result, returns 1 if x[i] is posititve or negative floating point NaN (Not a Number) and 0 otherwise.
    """
    ftype = impl.get_runtime().default_fp
    fx = ti.cast(x, ftype)
    if ti.static(ftype == ti.f64):
        y = ti.bit_cast(fx, ti.u64)
        return (ti.cast(y >> 32, ti.u32)
                & 0x7fffffff) + (ti.cast(y, ti.u32) != 0) > 0x7ff00000

    y = ti.bit_cast(fx, ti.u32)
    return (y & 0x7fffffff) > 0x7f800000


@ti.func
def vdir(ang):
    """Returns the 2d unit vector with argument equals `ang`.

    x (:mod:`~taichi.types.primitive_types`): The input angle in radians.

    Example:

        >>> @ti.kernel
        >>> def test():
        >>>     x = pi / 2
        >>>     print(ti.math.vdir(x))  # [0, 1]

    Returns:
        a 2d vector with argument equals `ang`.
    """
    return vec2(cos(ang), sin(ang))


__all__ = [
    "acos", "asin", "atan2", "ceil", "clamp", "cos", "cross", "degrees",
    "determinant", "distance", "dot", "e", "exp", "eye", "floor", "fract",
    "inf", "inverse", "isinf", "isnan", "ivec2", "ivec3", "ivec4", "length",
    "log", "log2", "mat2", "mat3", "mat4", "max", "min", "mix", "mod",
    "translate", "scale", "nan", "normalize", "pi", "pow", "radians",
    "reflect", "refract", "rot_by_axis", "rot_yaw_pitch_roll", "rotation2d",
    "rotation3d", "round", "sign", "sin", "smoothstep", "sqrt", "step", "tan",
    "tanh", "uvec2", "uvec3", "uvec4", "vdir", "vec2", "vec3", "vec4"
]
