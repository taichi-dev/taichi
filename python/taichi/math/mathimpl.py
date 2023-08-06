# pylint: disable=W0622
"""
Math functions for glsl-like functions and other stuff.
"""
import math

from taichi.lang import impl, ops
from taichi.lang.impl import static, zero
from taichi.lang.kernel_impl import func
from taichi.lang.matrix import Matrix
from taichi.lang.ops import (
    acos,
    asin,
    atan2,
    ceil,
    cos,
    exp,
    floor,
    log,
    max,
    min,
    pow,
    round,
    sin,
    sqrt,
    tan,
    tanh,
)
from taichi.types import matrix, template, vector
from taichi.types.primitive_types import f64, u32, u64

cfg = impl.default_cfg

e = math.e
"""The mathematical constant e = 2.718281….
Directly imported from the Python standard library `math`.
"""

pi = math.pi
"""The mathematical constant π = 3.141592….
Directly imported from the Python standard library `math`.
"""

inf = math.inf
"""A floating-point positive infinity. (For negative infinity, use `-inf`).
Directly imported from the Python standard library `math`.
"""

nan = math.nan
"""A floating-point "not a number" (NaN) value.
Directly imported from the Python standard library `math`
"""

vec2 = vector(2, cfg().default_fp)
"""2D floating vector type.
"""

vec3 = vector(3, cfg().default_fp)
"""3D floating vector type.
"""

vec4 = vector(4, cfg().default_fp)
"""4D floating vector type.
"""

ivec2 = vector(2, cfg().default_ip)
"""2D signed int vector type.
"""

ivec3 = vector(3, cfg().default_ip)
"""3D signed int vector type.
"""

ivec4 = vector(4, cfg().default_ip)
"""3D signed int vector type.
"""

uvec2 = vector(2, cfg().default_up)
"""2D unsigned int vector type.
"""

uvec3 = vector(3, cfg().default_up)
"""3D unsigned int vector type.
"""

uvec4 = vector(4, cfg().default_up)
"""4D unsigned int vector type.
"""

mat2 = matrix(2, 2, cfg().default_fp)
"""2x2 floating matrix type.
"""

mat3 = matrix(3, 3, cfg().default_fp)
"""3x3 floating matrix type.
"""

mat4 = matrix(4, 4, cfg().default_fp)
"""4x4 floating matrix type.
"""


@func
def mix(x, y, a):
    """Performs a linear interpolation between `x` and `y` using
    `a` to weight between them. The return value is computed as
    `x * (1 - a) + a * y`.

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
        >>> ti.math.mix(x, y, a)
        [2.000000, 1.000000, 1.000000]
        >>> x = ti.Matrix([[1, 2], [2, 3]], ti.f32)
        >>> y = ti.Matrix([[3, 5], [4, 5]], ti.f32)
        >>> a = 0.5
        >>> ti.math.mix(x, y, a)
        [[2.000000, 3.500000], [3.000000, 4.000000]]
    """
    return x * (1.0 - a) + y * a


@func
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
        >>> ti.math.clamp(v, 0.5, 1.0)
        [0.500000, 0.500000, 1.000000, 1.000000]
        >>> x = ti.Matrix([[0, 1], [-2, 2]], ti.f32)
        >>> y = ti.Matrix([[1, 2], [1, 2]], ti.f32)
        >>> ti.math.clamp(x, 0.5, y)
        [[0.500000, 1.000000], [0.500000, 2.000000]]
    """
    return ops.min(xmax, ops.max(xmin, x))


@func
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
        >>> ti.math.step(x, y)
        [[1.000000, 1.000000], [0.000000, 0.000000]]
    """
    return ops.cast(x >= edge, float)


@func
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
        >>> ti.math.fract(x)
        [0.800000, 0.300000, 0.300000, 0.200000]
    """
    return x - ops.floor(x)


@func
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
        >>> ti.math.smoothstep(edge0, edge1, x)
        [0.500000, 1.000000, 0.000000]
    """
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@func
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
        >>> ti.math.sign(x)
        [-1.000000, 0.000000, 1.000000]
    """
    return ops.cast((x >= 0.0), float) - ops.cast((x <= 0.0), float)


@func
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
        >>> ti.math.normalize(v)
        [0.267261, 0.534522, 0.801784]
    """
    return v / v.norm()


@func
def log2(x):
    """Return the base 2 logarithm of `x`, so that if :math:`2^y=x`,
    then :math:`y=\\log2(x)`.

    This is equivalent to the `log2` function is GLSL.

    Args:
        x (:class:`~taichi.Matrix`): The input value.

    Returns:
        The base 2 logarithm of `x`.

    Example::

        >>> x = ti.Vector([1., 2., 3.])
        >>> ti.math.log2(x)
        [0.000000, 1.000000, 1.584962]
    """
    return ops.log(x) / static(ops.log(2.0))


@func
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
        >>> ti.math.reflect(x, n)
        [1.000000, -2.000000, 3.000000]
    """
    k = x.dot(n)
    return x - 2.0 * k * n


@func
def degrees(x):
    """Convert `x` in radians to degrees, element-wise.

    Args:
        x (:class:`~taichi.Matrix`): The input angle in radians.

    Returns:
        angle in degrees.

    Example::

        >>> x = ti.Vector([-pi/2, pi/2])
        >>> ti.math.degrees(x)
        [-90.000000, 90.000000]
    """
    return x * static(180.0 / pi)


@func
def radians(x):
    """Convert `x` in degrees to radians, element-wise.

    Args:
        x (:class:`~taichi.Matrix`): The input angle in degrees.

    Returns:
        angle in radians.

    Example::

        >>> x = ti.Vector([-90., 45., 90.])
        >>> ti.math.radians(x) / pi
        [-0.500000, 0.250000, 0.500000]
    """
    return x * static(pi / 180.0)


@func
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
        >>> ti.math.distance(x, y)
        1.732051
    """
    return (x - y).norm()


@func
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
        >>> y = ti.Vector([0, 1., 0])
        >>> ti.math.refract(x, y, 2.0)
        [2.000000, -1.000000, 2.000000]
    """
    dxn = x.dot(n)
    result = zero(x)
    k = 1.0 - eta * eta * (1.0 - dxn * dxn)
    if k >= 0.0:
        result = eta * x - (eta * dxn + ops.sqrt(k)) * n
    return result


@func
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
        >>> ti.math.dot(x, y)
        1.000000
    """
    return x.dot(y)


@func
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
        >>> ti.math.cross(x, y)
        [0.000000, 0.000000, 1.000000]
    """
    return x.cross(y)


@func
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
        >>> ti.math.mod(x, y)
        [0.500000, 0.500000, 0.000000]
    """
    return x - y * ops.floor(x / y)


@func
def translate(dx, dy, dz):
    """Constructs a translation Matrix with shape (4, 4).

    Args:
        dx (float): delta x.
        dy (float): delta y.
        dz (float): delta z.

    Returns:
        :class:`~taichi.math.mat4`: translation matrix.

    Example::

        >>> ti.math.translate(1, 2, 3)
        [[ 1. 0. 0. 1.]
         [ 0. 1. 0. 2.]
         [ 0. 0. 1. 3.]
         [ 0. 0. 0. 1.]]
    """
    return mat4(
        [
            [1.0, 0.0, 0.0, dx],
            [0.0, 1.0, 0.0, dy],
            [0.0, 0.0, 1.0, dz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@func
def scale(sx, sy, sz):
    """Constructs a scale Matrix with shape (4, 4).

    Args:
        sx (float): scale x.
        sy (float): scale y.
        sz (float): scale z.

    Returns:
        :class:`~taichi.math.mat4`: scale matrix.

    Example::

        >>> ti.math.scale(1, 2, 3)
        [[ 1. 0. 0. 0.]
         [ 0. 2. 0. 0.]
         [ 0. 0. 3. 0.]
         [ 0. 0. 0. 1.]]
    """
    return mat4(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@func
def rot_by_axis(axis, ang):
    """Returns the 4x4 matrix representation of a 3d rotation with given axis `axis` and angle `ang`.

    Args:
        axis (vec3): rotation axis
        ang (float): angle in radians unit

    Returns:
        :class:`~taichi.math.mat4`: rotation matrix
    """
    c = ops.cos(ang)
    s = ops.sin(ang)

    axis = normalize(axis)
    temp = (1 - c) * axis
    return mat4(
        [
            [
                c + temp[0] * axis[0],
                temp[0] * axis[1] + s * axis[2],
                temp[0] * axis[2] - s * axis[1],
                0.0,
            ],
            [
                temp[1] * axis[0] - s * axis[2],
                c + temp[1] * axis[1],
                temp[1] * axis[2] + s * axis[0],
                0.0,
            ],
            [
                temp[2] * axis[0] + s * axis[1],
                temp[2] * axis[1] - s * axis[0],
                c + temp[2] * axis[2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@func
def rot_yaw_pitch_roll(yaw, pitch, roll):
    """Returns a 4x4 homogeneous rotation matrix representing the 3d rotation with Euler angles (rotate with Y axis first, X axis second, Z axis third).

    Args:
        yaw   (float): yaw angle in radians unit
        pitch (float): pitch angle in radians unit
        roll  (float): roll angle in radians unit

    Returns:
        :class:`~taichi.math.mat4`: rotation matrix
    """
    ch = ops.cos(yaw)
    sh = ops.sin(yaw)
    cp = ops.cos(pitch)
    sp = ops.sin(pitch)
    cb = ops.cos(roll)
    sb = ops.sin(roll)

    return mat4(
        [
            [ch * cb + sh * sp * sb, sb * cp, -sh * cb + ch * sp * sb, 0.0],
            [-ch * sb + sh * sp * cb, cb * cp, sb * sh + ch * sp * cb, 0.0],
            [sh * cp, -sp, ch * cp, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@func
def rotation2d(ang):
    """Returns the matrix representation of a 2d counter-clockwise rotation,
    given the angle of rotation.

    Args:
        ang (float): Angle of rotation in radians.

    Returns:
        :class:`~taichi.math.mat2`: 2x2 rotation matrix.

    Example::

        >>>ti.math.rotation2d(ti.math.radians(30))
        [[0.866025, -0.500000], [0.500000, 0.866025]]
    """
    ca, sa = ops.cos(ang), ops.sin(ang)
    return mat2([[ca, -sa], [sa, ca]])


@func
def rotation3d(ang_x, ang_y, ang_z):
    """Returns a 4x4 homogeneous rotation matrix representing the 3d rotation with Euler angles (rotate with Y axis first, X axis second, Z axis third).

    Args:
        ang_x (float): angle in radians unit around X axis
        ang_y (float): angle in radians unit around Y axis
        ang_z (float): angle in radians unit around Z axis
    Returns:
        :class:`~taichi.math.mat4`: rotation matrix
    Example:

        >>> ti.math.rotation3d(0.52, -0.785, 1.046)
        [[ 0.05048351 -0.61339645 -0.78816002  0.        ]
        [ 0.65833154  0.61388511 -0.4355969   0.        ]
        [ 0.75103329 -0.49688014  0.4348093   0.        ]
        [ 0.          0.          0.          1.        ]]
    """
    return rot_yaw_pitch_roll(ang_z, ang_x, ang_y)


@func
def eye(n: template()):
    """Returns the nxn identity matrix.

    Alias for :func:`~taichi.Matrix.identity`.
    """
    return Matrix.identity(float, n)


@func
def length(x):
    """Calculate the length of a vector.

    This function is equivalent to the `length` function in GLSL.
    Args:
        x (:class:`~taichi.Matrix`): The vector of which to calculate the length.

    Returns:
        The Euclidean norm of the vector.

    Example::

        >>> x = ti.Vector([1, 1, 1])
        >>> ti.math.length(x)
        1.732051
    """
    return x.norm()


@func
def determinant(m):
    """Alias for :func:`taichi.Matrix.determinant`."""
    return m.determinant()


@func
def inverse(mat):  # pylint: disable=R1710
    """Calculate the inverse of a matrix.

    This function is equivalent to the `inverse` function in GLSL.

    Args:
        mat (:class:`taichi.Matrix`): The matrix of which to take the inverse. \
            Supports only 2x2, 3x3 and 4x4 matrices.

    Returns:
        Inverse of the input matrix.

    Example::

        >>> m = ti.math.mat3([(1, 1, 0), (0, 1, 1), (0, 0, 1)])
        >>> ti.math.inverse(m)
        [[1.000000, -1.000000, 1.000000],
         [0.000000, 1.000000, -1.000000],
         [0.000000, 0.000000, 1.000000]]
    """
    return mat.inverse()


@func
def isinf(x):
    """Determines whether the parameter is positive or negative infinity, element-wise.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`taichi.Matrix`): The input.

    Example:

       >>> x = ti.math.vec4(inf, -inf, nan, 1)
       >>> ti.math.isinf(x)
       [1, 1, 0, 0]

    Returns:
        For each element i of the result, returns 1 if x[i] is posititve or negative floating point infinity and 0 otherwise.
    """
    ftype = impl.get_runtime().default_fp
    fx = ops.cast(x, ftype)
    if static(ftype == f64):
        y = ops.bit_cast(fx, u64)
        return (ops.cast(y >> 32, u32) & 0x7FFFFFFF) == 0x7FF00000 and (ops.cast(y, u32) == 0)

    y = ops.bit_cast(fx, u32)
    return (y & 0x7FFFFFFF) == 0x7F800000


@func
def isnan(x):
    """Determines whether the parameter is a number, element-wise.

    Args:
        x (:mod:`~taichi.types.primitive_types`, :class:`taichi.Matrix`): The input.

    Example:

       >>> x = ti.math.vec4(nan, -nan, inf, 1)
       >>> ti.math.isnan(x)
       [1, 1, 0, 0]

    Returns:
        For each element i of the result, returns 1 if x[i] is posititve or negative floating point NaN (Not a Number) and 0 otherwise.
    """
    ftype = impl.get_runtime().default_fp
    fx = ops.cast(x, ftype)
    if static(ftype == f64):
        y = ops.bit_cast(fx, u64)
        return (ops.cast(y >> 32, u32) & 0x7FFFFFFF) + (ops.cast(y, u32) != 0) > 0x7FF00000

    y = ops.bit_cast(fx, u32)
    return (y & 0x7FFFFFFF) > 0x7F800000


@func
def vdir(ang):
    """Returns the 2d unit vector with argument equals `ang`.

    x (:mod:`~taichi.types.primitive_types`): The input angle in radians.

    Example:

        >>> x = pi / 2
        >>> ti.math.vdir(x)
        [0, 1]

    Returns:
        a 2d vector with argument equals `ang`.
    """
    return vec2(cos(ang), sin(ang))


@func
def popcnt(x):
    return ops.popcnt(x)


@func
def clz(x):
    return ops.clz(x)


__all__ = [
    "acos",
    "asin",
    "atan2",
    "ceil",
    "clamp",
    "clz",
    "cos",
    "cross",
    "degrees",
    "determinant",
    "distance",
    "dot",
    "e",
    "exp",
    "eye",
    "floor",
    "fract",
    "inf",
    "inverse",
    "isinf",
    "isnan",
    "ivec2",
    "ivec3",
    "ivec4",
    "length",
    "log",
    "log2",
    "mat2",
    "mat3",
    "mat4",
    "max",
    "min",
    "mix",
    "mod",
    "translate",
    "scale",
    "nan",
    "normalize",
    "pi",
    "pow",
    "popcnt",
    "radians",
    "reflect",
    "refract",
    "rot_by_axis",
    "rot_yaw_pitch_roll",
    "rotation2d",
    "rotation3d",
    "round",
    "sign",
    "sin",
    "smoothstep",
    "sqrt",
    "step",
    "tan",
    "tanh",
    "uvec2",
    "uvec3",
    "uvec4",
    "vdir",
    "vec2",
    "vec3",
    "vec4",
]
