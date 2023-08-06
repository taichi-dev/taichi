import builtins
import functools
import operator as _bt_ops_mod  # bt for builtin
from typing import Union

import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import expr, impl
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.field import Field
from taichi.lang.util import cook_dtype, is_matrix_class, is_taichi_class, taichi_scope


def stack_info():
    return impl.get_runtime().get_current_src_info()


def is_taichi_expr(a):
    return isinstance(a, expr.Expr)


def wrap_if_not_expr(a):
    return (
        expr.Expr(a, dbg_info=_ti_core.DebugInfo(impl.get_runtime().get_current_src_info()))
        if not is_taichi_expr(a)
        else a
    )


def _read_matrix_or_scalar(x):
    if is_matrix_class(x):
        return x.to_numpy()
    return x


def writeback_binary(foo):
    @functools.wraps(foo)
    def wrapped(a, b):
        if isinstance(a, Field) or isinstance(b, Field):
            return NotImplemented
        if not (is_taichi_expr(a) and a.ptr.is_lvalue()):
            raise TaichiSyntaxError(f"cannot use a non-writable target as the first operand of '{foo.__name__}'")
        return foo(a, wrap_if_not_expr(b))

    return wrapped


def cast(obj, dtype):
    """Copy and cast a scalar or a matrix to a specified data type.
    Must be called in Taichi scope.

    Args:
        obj (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

        dtype (:mod:`~taichi.types.primitive_types`): A primitive type defined in :mod:`~taichi.types.primitive_types`.

    Returns:
        A copy of `obj`, casted to the specified data type `dtype`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([0, 1, 2], ti.i32)
        >>>     y = ti.cast(x, ti.f32)
        >>>     print(y)
        >>>
        >>> test()
        [0.0, 1.0, 2.0]
    """
    dtype = cook_dtype(dtype)
    if is_taichi_class(obj):
        # TODO: unify with element_wise_unary
        return obj.cast(dtype)
    return expr.Expr(_ti_core.value_cast(expr.Expr(obj).ptr, dtype))


def bit_cast(obj, dtype):
    """Copy and cast a scalar to a specified data type with its underlying
    bits preserved. Must be called in taichi scope.

    This function is equivalent to `reinterpret_cast` in C++.

    Args:
        obj (:mod:`~taichi.types.primitive_types`): Input scalar.

        dtype (:mod:`~taichi.types.primitive_types`): Target data type, must have \
            the same precision bits as the input (hence `f32` -> `f64` is not allowed).

    Returns:
        A copy of `obj`, casted to the specified data type `dtype`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = 3.14
        >>>     y = ti.bit_cast(x, ti.i32)
        >>>     print(y)  # 1078523331
        >>>
        >>>     z = ti.bit_cast(y, ti.f32)
        >>>     print(z)  # 3.14
    """
    dtype = cook_dtype(dtype)
    if is_taichi_class(obj):
        raise ValueError("Cannot apply bit_cast on Taichi classes")
    else:
        return expr.Expr(_ti_core.bits_cast(expr.Expr(obj).ptr, dtype))


def _unary_operation(taichi_op, python_op, a):
    if isinstance(a, Field):
        return NotImplemented
    if is_taichi_expr(a):
        return expr.Expr(taichi_op(a.ptr), dbg_info=_ti_core.DebugInfo(stack_info()))
    from taichi.lang.matrix import Matrix  # pylint: disable-msg=C0415

    if isinstance(a, Matrix):
        return Matrix(python_op(a.to_numpy()))
    return python_op(a)


def _binary_operation(taichi_op, python_op, a, b):
    if isinstance(a, Field) or isinstance(b, Field):
        return NotImplemented
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return expr.Expr(taichi_op(a.ptr, b.ptr), dbg_info=_ti_core.DebugInfo(stack_info()))
    from taichi.lang.matrix import Matrix  # pylint: disable-msg=C0415

    if isinstance(a, Matrix) or isinstance(b, Matrix):
        return Matrix(python_op(_read_matrix_or_scalar(a), _read_matrix_or_scalar(b)))
    return python_op(a, b)


def _ternary_operation(taichi_op, python_op, a, b, c):
    if isinstance(a, Field) or isinstance(b, Field) or isinstance(c, Field):
        return NotImplemented
    if is_taichi_expr(a) or is_taichi_expr(b) or is_taichi_expr(c):
        a, b, c = wrap_if_not_expr(a), wrap_if_not_expr(b), wrap_if_not_expr(c)
        return expr.Expr(taichi_op(a.ptr, b.ptr, c.ptr), dbg_info=_ti_core.DebugInfo(stack_info()))
    from taichi.lang.matrix import Matrix  # pylint: disable-msg=C0415

    if isinstance(a, Matrix) or isinstance(b, Matrix) or isinstance(c, Matrix):
        return Matrix(
            python_op(
                _read_matrix_or_scalar(a),
                _read_matrix_or_scalar(b),
                _read_matrix_or_scalar(c),
            )
        )
    return python_op(a, b, c)


def neg(x):
    """Numerical negative, element-wise.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

    Returns:
        Matrix or scalar `y`, so that `y = -x`. `y` has the same type as `x`.

    Example::
        >>> x = ti.Matrix([1, -1])
        >>> y = ti.neg(a)
        >>> y
        [-1, 1]
    """
    return _unary_operation(_ti_core.expr_neg, _bt_ops_mod.neg, x)


def sin(x):
    """Trigonometric sine, element-wise.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Angle, in radians.

    Returns:
        The sine of each element of `x`.

    Example::

        >>> from math import pi
        >>> x = ti.Matrix([-pi/2., 0, pi/2.])
        >>> ti.sin(x)
        [-1., 0., 1.]
    """
    return _unary_operation(_ti_core.expr_sin, np.sin, x)


def cos(x):
    """Trigonometric cosine, element-wise.

    Args:
        x (Union[:mod:`~taichi.type.primitive_types`, :class:`~taichi.Matrix`]): \
            Angle, in radians.

    Returns:
        The cosine of each element of `x`.

    Example::

        >>> from math import pi
        >>> x = ti.Matrix([-pi, 0, pi/2.])
        >>> ti.cos(x)
        [-1., 1., 0.]
    """
    return _unary_operation(_ti_core.expr_cos, np.cos, x)


def asin(x):
    """Trigonometric inverse sine, element-wise.

    The inverse of `sin` so that, if `y = sin(x)`, then `x = asin(y)`.

    For input `x` not in the domain `[-1, 1]`, this function returns `nan` if \
        it's called in taichi scope, or raises exception if it's called in python scope.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            A scalar or a matrix with elements in [-1, 1].

    Returns:
        The inverse sine of each element in `x`, in radians and in the closed \
            interval `[-pi/2, pi/2]`.

    Example::

        >>> from math import pi
        >>> ti.asin(ti.Matrix([-1.0, 0.0, 1.0])) * 180 / pi
        [-90., 0., 90.]
    """
    return _unary_operation(_ti_core.expr_asin, np.arcsin, x)


def acos(x):
    """Trigonometric inverse cosine, element-wise.

    The inverse of `cos` so that, if `y = cos(x)`, then `x = acos(y)`.

    For input `x` not in the domain `[-1, 1]`, this function returns `nan` if \
        it's called in taichi scope, or raises exception if it's called in python scope.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            A scalar or a matrix with elements in [-1, 1].

    Returns:
        The inverse cosine of each element in `x`, in radians and in the closed \
            interval `[0, pi]`. This is a scalar if `x` is a scalar.

    Example::

        >>> from math import pi
        >>> ti.acos(ti.Matrix([-1.0, 0.0, 1.0])) * 180 / pi
        [180., 90., 0.]
    """
    return _unary_operation(_ti_core.expr_acos, np.arccos, x)


def sqrt(x):
    """Return the non-negative square-root of a scalar or a matrix,
    element wise. If `x < 0` an exception is raised.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The scalar or matrix whose square-roots are required.

    Returns:
        The square-root `y` so that `y >= 0` and `y^2 = x`. `y` has the same type as `x`.

    Example::

        >>> x = ti.Matrix([1., 4., 9.])
        >>> y = ti.sqrt(x)
        >>> y
        [1.0, 2.0, 3.0]
    """
    return _unary_operation(_ti_core.expr_sqrt, np.sqrt, x)


def rsqrt(x):
    """The reciprocal of the square root function.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            A scalar or a matrix.

    Returns:
        The reciprocal of `sqrt(x)`.
    """

    def _rsqrt(x):
        return 1 / np.sqrt(x)

    return _unary_operation(_ti_core.expr_rsqrt, _rsqrt, x)


def _round(x):
    return _unary_operation(_ti_core.expr_round, np.round, x)


def round(x, dtype=None):  # pylint: disable=redefined-builtin
    """Round to the nearest integer, element-wise.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            A scalar or a matrix.

        dtype: (:mod:`~taichi.types.primitive_types`): the returned type, default to `None`. If \
            set to `None` the retuned value will have the same type with `x`.

    Returns:
        The nearest integer of `x`, with return value type `dtype`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([-1.5, 1.2, 2.7])
        >>>     print(ti.round(x))
        [-2., 1., 3.]
    """
    result = _round(x)
    if dtype is not None:
        result = cast(result, dtype)
    return result


def _floor(x):
    return _unary_operation(_ti_core.expr_floor, np.floor, x)


def floor(x, dtype=None):
    """Return the floor of the input, element-wise.
    The floor of the scalar `x` is the largest integer `k`, such that `k <= x`.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

        dtype: (:mod:`~taichi.types.primitive_types`): the returned type, default to `None`. If \
            set to `None` the retuned value will have the same type with `x`.

    Returns:
        The floor of each element in `x`, with return value type `dtype`.

    Example::
        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([-1.1, 2.2, 3.])
        >>>     y = ti.floor(x, ti.f64)
        >>>     print(y)  # [-2.000000000000, 2.000000000000, 3.000000000000]
    """
    result = _floor(x)
    if dtype is not None:
        result = cast(result, dtype)
    return result


def _ceil(x):
    return _unary_operation(_ti_core.expr_ceil, np.ceil, x)


def ceil(x, dtype=None):
    """Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `k`, such that `k >= x`.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

        dtype: (:mod:`~taichi.types.primitive_types`): the returned type, default to `None`. If \
            set to `None` the retuned value will have the same type with `x`.

    Returns:
        The ceiling of each element in `x`, with return value type `dtype`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([3.14, -1.5])
        >>>     y = ti.ceil(x)
        >>>     print(y)  # [4.0, -1.0]
    """
    result = _ceil(x)
    if dtype is not None:
        result = cast(result, dtype)
    return result


def frexp(x):
    return _unary_operation(_ti_core.expr_frexp, np.frexp, x)


def tan(x):
    """Trigonometric tangent function, element-wise.

    Equivalent to `ti.sin(x)/ti.cos(x)` element-wise.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

    Returns:
        The tangent values of `x`.

    Example::

        >>> from math import pi
        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([-pi, pi/2, pi])
        >>>     y = ti.tan(x)
        >>>     print(y)
        >>>
        >>> test()
        [-0.0, -22877334.0, 0.0]
    """
    return _unary_operation(_ti_core.expr_tan, np.tan, x)


def tanh(x):
    """Compute the hyperbolic tangent of `x`, element-wise.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

    Returns:
        The corresponding hyperbolic tangent values.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([-1.0, 0.0, 1.0])
        >>>     y = ti.tanh(x)
        >>>     print(y)
        >>>
        >>> test()
        [-0.761594, 0.000000, 0.761594]
    """
    return _unary_operation(_ti_core.expr_tanh, np.tanh, x)


def exp(x):
    """Compute the exponential of all elements in `x`, element-wise.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

    Returns:
        Element-wise exponential of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([-1.0, 0.0, 1.0])
        >>>     y = ti.exp(x)
        >>>     print(y)
        >>>
        >>> test()
        [0.367879, 1.000000, 2.718282]
    """
    return _unary_operation(_ti_core.expr_exp, np.exp, x)


def log(x):
    """Compute the natural logarithm, element-wise.

    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base `e`.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

    Returns:
        The natural logarithm of `x`, element-wise.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([-1.0, 0.0, 1.0])
        >>>     y = ti.log(x)
        >>>     print(y)
        >>>
        >>> test()
        [-nan, -inf, 0.000000]
    """
    return _unary_operation(_ti_core.expr_log, np.log, x)


def abs(x):  # pylint: disable=W0622
    """Compute the absolute value :math:`|x|` of `x`, element-wise.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

    Returns:
        The absolute value of each element in `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([-1.0, 0.0, 1.0])
        >>>     y = ti.abs(x)
        >>>     print(y)
        >>>
        >>> test()
        [1.0, 0.0, 1.0]
    """
    return _unary_operation(_ti_core.expr_abs, builtins.abs, x)


def bit_not(a):
    """The bit not function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        Bitwise not of `a`.
    """
    return _unary_operation(_ti_core.expr_bit_not, _bt_ops_mod.invert, a)


def popcnt(a):
    def _popcnt(x):
        return bin(x).count("1")

    return _unary_operation(_ti_core.expr_popcnt, _popcnt, a)


def logical_not(a):
    """The logical not function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        `1` iff `a=0`, otherwise `0`.
    """
    return _unary_operation(_ti_core.expr_logic_not, np.logical_not, a)


def random(dtype=float) -> Union[float, int]:
    """Return a single random float/integer according to the specified data type.
    Must be called in taichi scope.

    If the required `dtype` is float type, this function returns a random number
    sampled from the uniform distribution in the half-open interval [0, 1).

    For integer types this function returns a random integer in the
    half-open interval [0, 2^32) if a 32-bit integer is required,
    or a random integer in the half-open interval [0, 2^64) if a
    64-bit integer is required.

    Args:
        dtype (:mod:`~taichi.types.primitive_types`): Type of the required random value.

    Returns:
        A random value with type `dtype`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.random(float)
        >>>     print(x)  # 0.090257
        >>>
        >>>     y = ti.random(ti.f64)
        >>>     print(y)  # 0.716101627301
        >>>
        >>>     i = ti.random(ti.i32)
        >>>     print(i)  # -963722261
        >>>
        >>>     j = ti.random(ti.i64)
        >>>     print(j)  # 73412986184350777
    """
    dtype = cook_dtype(dtype)
    x = expr.Expr(_ti_core.make_rand_expr(dtype, _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())))
    return impl.expr_init(x)


# NEXT: add matpow(self, power)


def add(a, b):
    """The add function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        sum of `a` and `b`.
    """
    return _binary_operation(_ti_core.expr_add, _bt_ops_mod.add, a, b)


def sub(a, b):
    """The sub function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        `a` subtract `b`.
    """
    return _binary_operation(_ti_core.expr_sub, _bt_ops_mod.sub, a, b)


def mul(a, b):
    """The multiply function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        `a` multiplied by `b`.
    """
    return _binary_operation(_ti_core.expr_mul, _bt_ops_mod.mul, a, b)


def mod(x1, x2):
    """Returns the element-wise remainder of division.

    This is equivalent to the Python modulus operator `x1 % x2` and
    has the same sign as the divisor x2.

    Args:
        x1 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Dividend scalar or matrix.

        x2 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Divisor scalar or matrix. When both `x1` and `x2` are matrices they must have the same shape.

    Returns:
        The element-wise remainder of the quotient `floordiv(x1, x2)`. This is a scalar \
            if both `x1` and `x2` are scalars.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([3.0, 4.0, 5.0])
        >>>     y = 3
        >>>     z = ti.mod(y, x)
        >>>     print(z)
        >>>
        >>> test()
        [1.0, 0.0, 4.0]
    """

    def expr_python_mod(a, b):
        # a % b = a - (a // b) * b
        quotient = expr.Expr(_ti_core.expr_floordiv(a, b))
        multiply = expr.Expr(_ti_core.expr_mul(b, quotient.ptr))
        return _ti_core.expr_sub(a, multiply.ptr)

    return _binary_operation(expr_python_mod, _bt_ops_mod.mod, x1, x2)


def pow(base, exponent):  # pylint: disable=W0622
    """First array elements raised to second array elements :math:`{base}^{exponent}`, element-wise.

    The result type of two scalar operands is determined as follows:
    - If the exponent is an integral value, then the result type takes the type of the base.
    - Otherwise, the result type follows
      [Implicit type casting in binary operations](https://docs.taichi-lang.org/docs/type#implicit-type-casting-in-binary-operations).

    With the above rules, an integral value raised to a negative integral value cannot have a
    feasible type. Therefore, an exception will be raised if debug mode or optimization passes
    are on; otherwise 1 will be returned.

    In the following situations, the result is undefined:
    - A negative value raised to a non-integral value.
    - A zero value raised to a non-positive value.

    Args:
        base (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The bases.
        exponent (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The exponents.

    Returns:
        `base` raised to `exponent`. This is a scalar if both `base` and `exponent` are scalars.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([-2.0, 2.0])
        >>>     y = -3
        >>>     z = ti.pow(x, y)
        >>>     print(z)
        >>>
        >>> test()
        [-0.125000, 0.125000]
    """
    return _binary_operation(_ti_core.expr_pow, _bt_ops_mod.pow, base, exponent)


def floordiv(a, b):
    """The floor division function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix with elements not equal to zero.

    Returns:
        The floor function of `a` divided by `b`.
    """
    return _binary_operation(_ti_core.expr_floordiv, _bt_ops_mod.floordiv, a, b)


def truediv(a, b):
    """True division function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix with elements not equal to zero.

    Returns:
        The true value of `a` divided by `b`.
    """
    return _binary_operation(_ti_core.expr_truediv, _bt_ops_mod.truediv, a, b)


def max_impl(a, b):
    """The maxnimum function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        The maxnimum of `a` and `b`.
    """
    return _binary_operation(_ti_core.expr_max, np.maximum, a, b)


def min_impl(a, b):
    """The minimum function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        The minimum of `a` and `b`.
    """
    return _binary_operation(_ti_core.expr_min, np.minimum, a, b)


def atan2(x1, x2):
    """Element-wise arc tangent of `x1/x2`.

    Args:
        x1 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            y-coordinates.
        x2 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            x-coordinates.

    Returns:
        Angles in radians, in the range `[-pi, pi]`.
        This is a scalar if both `x1` and `x2` are scalars.

    Example::

        >>> from math import pi
        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([-1.0, 1.0, -1.0, 1.0])
        >>>     y = ti.Matrix([-1.0, -1.0, 1.0, 1.0])
        >>>     z = ti.atan2(y, x) * 180 / pi
        >>>     print(z)
        >>>
        >>> test()
        [-135.0, -45.0, 135.0, 45.0]
    """
    return _binary_operation(_ti_core.expr_atan2, np.arctan2, x1, x2)


def raw_div(x1, x2):
    """Return `x1 // x2` if both `x1`, `x2` are integers, otherwise return `x1/x2`.

    Args:
        x1 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): Dividend.
        x2 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): Divisor.

    Returns:
        Return `x1 // x2` if both `x1`, `x2` are integers, otherwise return `x1/x2`.

    Example::

        >>> @ti.kernel
        >>> def main():
        >>>     x = 5
        >>>     y = 3
        >>>     print(raw_div(x, y))  # 1
        >>>     z = 4.0
        >>>     print(raw_div(x, z))  # 1.25
    """

    def c_div(a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a // b
        return a / b

    return _binary_operation(_ti_core.expr_div, c_div, x1, x2)


def raw_mod(x1, x2):
    """Return the remainder of `x1/x2`, element-wise.
    This is the C-style `mod` function.

    Args:
        x1 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The dividend.
        x2 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The divisor.

    Returns:
        The remainder of `x1` divided by `x2`.

    Example::

        >>> @ti.kernel
        >>> def main():
        >>>     print(ti.mod(-4, 3))  # 2
        >>>     print(ti.raw_mod(-4, 3))  # -1
    """

    def c_mod(x, y):
        return x - y * int(float(x) / y)

    return _binary_operation(_ti_core.expr_mod, c_mod, x1, x2)


def cmp_lt(a, b):
    """Compare two values (less than)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is strictly smaller than RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_lt, _bt_ops_mod.lt, a, b)


def cmp_le(a, b):
    """Compare two values (less than or equal to)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is smaller than or equal to RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_le, _bt_ops_mod.le, a, b)


def cmp_gt(a, b):
    """Compare two values (greater than)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is strictly larger than RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_gt, _bt_ops_mod.gt, a, b)


def cmp_ge(a, b):
    """Compare two values (greater than or equal to)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        bool: True if LHS is greater than or equal to RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_ge, _bt_ops_mod.ge, a, b)


def cmp_eq(a, b):
    """Compare two values (equal to)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is equal to RHS, False otherwise.

    """
    return _binary_operation(_ti_core.expr_cmp_eq, _bt_ops_mod.eq, a, b)


def cmp_ne(a, b):
    """Compare two values (not equal to)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is not equal to RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_ne, _bt_ops_mod.ne, a, b)


def bit_or(a, b):
    """Computes bitwise-or

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: LHS bitwise-or with RHS

    """
    return _binary_operation(_ti_core.expr_bit_or, _bt_ops_mod.or_, a, b)


def bit_and(a, b):
    """Compute bitwise-and

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: LHS bitwise-and with RHS

    """
    return _binary_operation(_ti_core.expr_bit_and, _bt_ops_mod.and_, a, b)


def bit_xor(a, b):
    """Compute bitwise-xor

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: LHS bitwise-xor with RHS

    """
    return _binary_operation(_ti_core.expr_bit_xor, _bt_ops_mod.xor, a, b)


def bit_shl(a, b):
    """Compute bitwise shift left

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, int]: LHS << RHS

    """
    return _binary_operation(_ti_core.expr_bit_shl, _bt_ops_mod.lshift, a, b)


def bit_sar(a, b):
    """Compute bitwise shift right

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, int]: LHS >> RHS

    """
    return _binary_operation(_ti_core.expr_bit_sar, _bt_ops_mod.rshift, a, b)


@taichi_scope
def bit_shr(x1, x2):
    """Elements in `x1` shifted to the right by number of bits in `x2`.
    Both `x1`, `x2` must have integer type.

    Args:
        x1 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input data.
        x2 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Number of bits to remove at the right of `x1`.

    Returns:
        Return `x1` with bits shifted `x2` times to the right.
        This is a scalar if both `x1` and `x2` are scalars.

    Example::
        >>> @ti.kernel
        >>> def main():
        >>>     x = ti.Matrix([7, 8])
        >>>     y = ti.Matrix([1, 2])
        >>>     print(ti.bit_shr(x, y))
        >>>
        >>> main()
        [3, 2]
    """
    return _binary_operation(_ti_core.expr_bit_shr, _bt_ops_mod.rshift, x1, x2)


def logical_and(a, b):
    """Compute logical_and

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: LHS logical-and RHS (with short-circuit semantics)

    """
    return _binary_operation(_ti_core.expr_logical_and, lambda a, b: a and b, a, b)


def logical_or(a, b):
    """Compute logical_or

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: LHS logical-or RHS (with short-circuit semantics)

    """
    return _binary_operation(_ti_core.expr_logical_or, lambda a, b: a or b, a, b)


def select(cond, x1, x2):
    """Return an array drawn from elements in `x1` or `x2`,
    depending on the conditions in `cond`.

    Args:
        cond (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The array of conditions.
        x1, x2 (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The arrays where the output elements are taken from.

    Returns:
        The output at position `k` is the k-th element of `x1` if the k-th element
        in `cond` is `True`, otherwise it's the k-th element of `x2`.

    Example::

        >>> @ti.kernel
        >>> def main():
        >>>     cond = ti.Matrix([0, 1, 0, 1])
        >>>     x = ti.Matrix([1, 2, 3, 4])
        >>>     y = ti.Matrix([-1, -2, -3, -4])
        >>>     print(ti.select(cond, x, y))
        >>>
        >>> main()
        [-1, 2, -3, 4]
    """
    # TODO: systematically resolve `-1 = True` problem by introducing u1:
    cond = logical_not(logical_not(cond))

    def py_select(cond, x1, x2):
        return x1 * cond + x2 * (1 - cond)

    return _ternary_operation(_ti_core.expr_select, py_select, cond, x1, x2)


def ifte(cond, x1, x2):
    """Evaluate and return `x1` if `cond` is true; otherwise evaluate and return `x2`. This operator guarantees
    short-circuit semantics: exactly one of `x1` or `x2` will be evaluated.

    Args:
        cond (:mod:`~taichi.types.primitive_types`): \
            The condition.
        x1, x2 (:mod:`~taichi.types.primitive_types`): \
            The outputs.

    Returns:
        `x1` if `cond` is true and `x2` otherwise.
    """
    # TODO: systematically resolve `-1 = True` problem by introducing u1:
    cond = logical_not(logical_not(cond))

    def py_ifte(cond, x1, x2):
        return x1 if cond else x2

    return _ternary_operation(_ti_core.expr_ifte, py_ifte, cond, x1, x2)


def clz(a):
    """Count the number of leading zeros for a 32bit integer"""

    def _clz(x):
        for i in range(32):
            if 2**i > x:
                return 32 - i
        return 0

    return _unary_operation(_ti_core.expr_clz, _clz, a)


@writeback_binary
def atomic_add(x, y):
    """Atomically compute `x + y`, store the result in `x`,
    and return the old value of `x`.

    `x` must be a writable target, constant expressions or scalars
    are not allowed.

    Args:
        x, y (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input.

    Returns:
        The old value of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([0, 0, 0])
        >>>     y = ti.Vector([1, 2, 3])
        >>>     z = ti.atomic_add(x, y)
        >>>     print(x)  # [1, 2, 3]  the new value of x
        >>>     print(z)  # [0, 0, 0], the old value of x
        >>>
        >>>     ti.atomic_add(1, x)  # will raise TaichiSyntaxError
    """
    return impl.expr_init(expr.Expr(_ti_core.expr_atomic_add(x.ptr, y.ptr), dbg_info=_ti_core.DebugInfo(stack_info())))


@writeback_binary
def atomic_mul(x, y):
    """Atomically compute `x * y`, store the result in `x`,
    and return the old value of `x`.

    `x` must be a writable target, constant expressions or scalars
    are not allowed.

    Args:
        x, y (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input.

    Returns:
        The old value of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([1, 2, 3])
        >>>     y = ti.Vector([4, 5, 6])
        >>>     z = ti.atomic_mul(x, y)
        >>>     print(x)  # [1, 2, 3]  the new value of x
        >>>     print(z)  # [4, 10, 18], the old value of x
        >>>
        >>>     ti.atomic_mul(1, x)  # will raise TaichiSyntaxError
    """
    return impl.expr_init(expr.Expr(_ti_core.expr_atomic_mul(x.ptr, y.ptr), dbg_info=_ti_core.DebugInfo(stack_info())))


@writeback_binary
def atomic_sub(x, y):
    """Atomically subtract `x` by `y`, store the result in `x`,
    and return the old value of `x`.

    `x` must be a writable target, constant expressions or scalars
    are not allowed.

    Args:
        x, y (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input.

    Returns:
        The old value of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([0, 0, 0])
        >>>     y = ti.Vector([1, 2, 3])
        >>>     z = ti.atomic_sub(x, y)
        >>>     print(x)  # [-1, -2, -3]  the new value of x
        >>>     print(z)  # [0, 0, 0], the old value of x
        >>>
        >>>     ti.atomic_sub(1, x)  # will raise TaichiSyntaxError
    """
    return impl.expr_init(expr.Expr(_ti_core.expr_atomic_sub(x.ptr, y.ptr), dbg_info=_ti_core.DebugInfo(stack_info())))


@writeback_binary
def atomic_min(x, y):
    """Atomically compute the minimum of `x` and `y`, element-wise.
    Store the result in `x`, and return the old value of `x`.

    `x` must be a writable target, constant expressions or scalars
    are not allowed.

    Args:
        x, y (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input.

    Returns:
        The old value of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = 2
        >>>     y = 1
        >>>     z = ti.atomic_min(x, y)
        >>>     print(x)  # 1  the new value of x
        >>>     print(z)  # 2, the old value of x
        >>>
        >>>     ti.atomic_min(1, x)  # will raise TaichiSyntaxError
    """
    return impl.expr_init(expr.Expr(_ti_core.expr_atomic_min(x.ptr, y.ptr), dbg_info=_ti_core.DebugInfo(stack_info())))


@writeback_binary
def atomic_max(x, y):
    """Atomically compute the maximum of `x` and `y`, element-wise.
    Store the result in `x`, and return the old value of `x`.

    `x` must be a writable target, constant expressions or scalars
    are not allowed.

    Args:
        x, y (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input.

    Returns:
        The old value of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = 1
        >>>     y = 2
        >>>     z = ti.atomic_max(x, y)
        >>>     print(x)  # 2  the new value of x
        >>>     print(z)  # 1, the old value of x
        >>>
        >>>     ti.atomic_max(1, x)  # will raise TaichiSyntaxError
    """
    return impl.expr_init(expr.Expr(_ti_core.expr_atomic_max(x.ptr, y.ptr), dbg_info=_ti_core.DebugInfo(stack_info())))


@writeback_binary
def atomic_and(x, y):
    """Atomically compute the bit-wise AND of `x` and `y`, element-wise.
    Store the result in `x`, and return the old value of `x`.

    `x` must be a writable target, constant expressions or scalars
    are not allowed.

    Args:
        x, y (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input. When both are matrices they must have the same shape.

    Returns:
        The old value of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([-1, 0, 1])
        >>>     y = ti.Vector([1, 2, 3])
        >>>     z = ti.atomic_and(x, y)
        >>>     print(x)  # [1, 0, 1]  the new value of x
        >>>     print(z)  # [-1, 0, 1], the old value of x
        >>>
        >>>     ti.atomic_and(1, x)  # will raise TaichiSyntaxError
    """
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_bit_and(x.ptr, y.ptr), dbg_info=_ti_core.DebugInfo(stack_info()))
    )


@writeback_binary
def atomic_or(x, y):
    """Atomically compute the bit-wise OR of `x` and `y`, element-wise.
    Store the result in `x`, and return the old value of `x`.

    `x` must be a writable target, constant expressions or scalars
    are not allowed.

    Args:
        x, y (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input. When both are matrices they must have the same shape.

    Returns:
        The old value of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([-1, 0, 1])
        >>>     y = ti.Vector([1, 2, 3])
        >>>     z = ti.atomic_or(x, y)
        >>>     print(x)  # [-1, 2, 3]  the new value of x
        >>>     print(z)  # [-1, 0, 1], the old value of x
        >>>
        >>>     ti.atomic_or(1, x)  # will raise TaichiSyntaxError
    """
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_bit_or(x.ptr, y.ptr), dbg_info=_ti_core.DebugInfo(stack_info()))
    )


@writeback_binary
def atomic_xor(x, y):
    """Atomically compute the bit-wise XOR of `x` and `y`, element-wise.
    Store the result in `x`, and return the old value of `x`.

    `x` must be a writable target, constant expressions or scalars
    are not allowed.

    Args:
        x, y (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input. When both are matrices they must have the same shape.

    Returns:
        The old value of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([-1, 0, 1])
        >>>     y = ti.Vector([1, 2, 3])
        >>>     z = ti.atomic_xor(x, y)
        >>>     print(x)  # [-2, 2, 2]  the new value of x
        >>>     print(z)  # [-1, 0, 1], the old value of x
        >>>
        >>>     ti.atomic_xor(1, x)  # will raise TaichiSyntaxError
    """
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_bit_xor(x.ptr, y.ptr), dbg_info=_ti_core.DebugInfo(stack_info()))
    )


@writeback_binary
def assign(a, b):
    impl.get_runtime().compiling_callable.ast_builder().expr_assign(a.ptr, b.ptr, _ti_core.DebugInfo(stack_info()))
    return a


def max(*args):  # pylint: disable=W0622
    """Compute the maximum of the arguments, element-wise.

    This function takes no effect on a single argument, even it's array-like.
    When there are both scalar and matrix arguments in `args`, the matrices
    must have the same shape, and scalars will be broadcasted to the same shape as the matrix.

    Args:
        args: (List[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input.

    Returns:
        Maximum of the inputs.

    Example::

        >>> @ti.kernel
        >>> def foo():
        >>>     x = ti.Vector([0, 1, 2])
        >>>     y = ti.Vector([3, 4, 5])
        >>>     z = ti.max(x, y, 4)
        >>>     print(z)  # [4, 4, 5]
    """
    num_args = len(args)
    assert num_args >= 1
    if num_args == 1:
        return args[0]
    if num_args == 2:
        return max_impl(args[0], args[1])
    return max_impl(args[0], max(*args[1:]))


def min(*args):  # pylint: disable=W0622
    """Compute the minimum of the arguments, element-wise.

    This function takes no effect on a single argument, even it's array-like.
    When there are both scalar and matrix arguments in `args`, the matrices
    must have the same shape, and scalars will be broadcasted to the same shape as the matrix.

    Args:
        args: (List[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            The input.

    Returns:
        Minimum of the inputs.

    Example::

        >>> @ti.kernel
        >>> def foo():
        >>>     x = ti.Vector([0, 1, 2])
        >>>     y = ti.Vector([3, 4, 5])
        >>>     z = ti.min(x, y, 1)
        >>>     print(z)  # [0, 1, 1]
    """
    num_args = len(args)
    assert num_args >= 1
    if num_args == 1:
        return args[0]
    if num_args == 2:
        return min_impl(args[0], args[1])
    return min_impl(args[0], min(*args[1:]))


__all__ = [
    "acos",
    "asin",
    "atan2",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "atomic_max",
    "atomic_sub",
    "atomic_min",
    "atomic_add",
    "atomic_mul",
    "bit_cast",
    "bit_shr",
    "cast",
    "ceil",
    "cos",
    "exp",
    "floor",
    "frexp",
    "log",
    "random",
    "raw_mod",
    "raw_div",
    "round",
    "rsqrt",
    "sin",
    "sqrt",
    "tan",
    "tanh",
    "max",
    "min",
    "select",
    "abs",
    "pow",
]
