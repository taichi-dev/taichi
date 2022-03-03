import builtins
import functools
import math
import operator as _bt_ops_mod  # bt for builtin
import traceback

from taichi._lib import core as _ti_core
from taichi.lang import expr, impl
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.util import cook_dtype, is_taichi_class, taichi_scope

unary_ops = []


def stack_info():
    s = traceback.extract_stack()[3:-1]
    for i, l in enumerate(s):
        if 'taichi_ast_generator' in l:
            s = s[i + 1:]
            break
    raw = ''.join(traceback.format_list(s))
    # remove the confusing last line
    return '\n'.join(raw.split('\n')[:-5]) + '\n'


def is_taichi_expr(a):
    return isinstance(a, expr.Expr)


def wrap_if_not_expr(a):
    return expr.Expr(a) if not is_taichi_expr(a) else a


def unary(foo):
    @functools.wraps(foo)
    def imp_foo(x):
        return foo(x)

    @functools.wraps(foo)
    def wrapped(a):
        if is_taichi_class(a):
            return a._element_wise_unary(imp_foo)
        return imp_foo(a)

    return wrapped


binary_ops = []


def binary(foo):
    @functools.wraps(foo)
    def imp_foo(x, y):
        return foo(x, y)

    @functools.wraps(foo)
    def rev_foo(x, y):
        return foo(y, x)

    @functools.wraps(foo)
    def wrapped(a, b):
        if is_taichi_class(a):
            return a._element_wise_binary(imp_foo, b)
        if is_taichi_class(b):
            return b._element_wise_binary(rev_foo, a)
        return imp_foo(a, b)

    binary_ops.append(wrapped)
    return wrapped


ternary_ops = []


def ternary(foo):
    @functools.wraps(foo)
    def abc_foo(a, b, c):
        return foo(a, b, c)

    @functools.wraps(foo)
    def bac_foo(b, a, c):
        return foo(a, b, c)

    @functools.wraps(foo)
    def cab_foo(c, a, b):
        return foo(a, b, c)

    @functools.wraps(foo)
    def wrapped(a, b, c):
        if is_taichi_class(a):
            return a._element_wise_ternary(abc_foo, b, c)
        if is_taichi_class(b):
            return b._element_wise_ternary(bac_foo, a, c)
        if is_taichi_class(c):
            return c._element_wise_ternary(cab_foo, a, b)
        return abc_foo(a, b, c)

    ternary_ops.append(wrapped)
    return wrapped


writeback_binary_ops = []


def writeback_binary(foo):
    @functools.wraps(foo)
    def imp_foo(x, y):
        return foo(x, wrap_if_not_expr(y))

    @functools.wraps(foo)
    def wrapped(a, b):
        if is_taichi_class(a):
            return a._element_wise_writeback_binary(imp_foo, b)
        if is_taichi_class(b):
            raise TaichiSyntaxError(
                f'cannot augassign taichi class {type(b)} to scalar expr')
        else:
            return imp_foo(a, b)

    writeback_binary_ops.append(wrapped)
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
        raise ValueError('Cannot apply bit_cast on Taichi classes')
    else:
        return expr.Expr(_ti_core.bits_cast(expr.Expr(obj).ptr, dtype))


def _unary_operation(taichi_op, python_op, a):
    if is_taichi_expr(a):
        return expr.Expr(taichi_op(a.ptr), tb=stack_info())
    return python_op(a)


def _binary_operation(taichi_op, python_op, a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return expr.Expr(taichi_op(a.ptr, b.ptr), tb=stack_info())
    return python_op(a, b)


def _ternary_operation(taichi_op, python_op, a, b, c):
    if is_taichi_expr(a) or is_taichi_expr(b) or is_taichi_expr(c):
        a, b, c = wrap_if_not_expr(a), wrap_if_not_expr(b), wrap_if_not_expr(c)
        return expr.Expr(taichi_op(a.ptr, b.ptr, c.ptr), tb=stack_info())
    return python_op(a, b, c)


@unary
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


@unary
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
    return _unary_operation(_ti_core.expr_sin, math.sin, x)


@unary
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
    return _unary_operation(_ti_core.expr_cos, math.cos, x)


@unary
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
    return _unary_operation(_ti_core.expr_asin, math.asin, x)


@unary
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
    return _unary_operation(_ti_core.expr_acos, math.acos, x)


@unary
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
    return _unary_operation(_ti_core.expr_sqrt, math.sqrt, x)


@unary
def rsqrt(x):
    """The reciprocal of the square root function.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            A scalar or a matrix.

    Returns:
        The reciprocal of `sqrt(x)`.
    """
    def _rsqrt(x):
        return 1 / math.sqrt(x)

    return _unary_operation(_ti_core.expr_rsqrt, _rsqrt, x)


@unary
def round(x):  # pylint: disable=redefined-builtin
    """Round to the nearest integer, element-wise.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            A scalar or a matrix.

    Returns:
        The nearest integer of `x`.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Vector([-1.5, 1.2, 2.7])
        >>>     print(ti.round(x))
        [-2., 1., 3.]
    """
    return _unary_operation(_ti_core.expr_round, builtins.round, x)


@unary
def floor(x):
    """Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `k`, such that `k <= x`.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

    Returns:
        The floor of each element in `x`, with float type.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([3.14, -1.5])
        >>>     y = ti.floor(x)
        >>>     print(y)  # [3.0, -2.0]
    """
    return _unary_operation(_ti_core.expr_floor, math.floor, x)


@unary
def ceil(x):
    """Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `k`, such that `k >= x`.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): \
            Input scalar or matrix.

    Returns:
        The ceiling of each element in `x`, with float dtype.

    Example::

        >>> @ti.kernel
        >>> def test():
        >>>     x = ti.Matrix([3.14, -1.5])
        >>>     y = ti.ceil(x)
        >>>     print(y)  # [4.0, -1.0]
    """
    return _unary_operation(_ti_core.expr_ceil, math.ceil, x)


@unary
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
    return _unary_operation(_ti_core.expr_tan, math.tan, x)


@unary
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
    return _unary_operation(_ti_core.expr_tanh, math.tanh, x)


@unary
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
    return _unary_operation(_ti_core.expr_exp, math.exp, x)


@unary
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
    return _unary_operation(_ti_core.expr_log, math.log, x)


@unary
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


@unary
def bit_not(a):
    """The bit not function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        Bitwise not of `a`.
    """
    return _unary_operation(_ti_core.expr_bit_not, _bt_ops_mod.invert, a)


@unary
def logical_not(a):
    """The logical not function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        `1` iff `a=0`, otherwise `0`.
    """
    return _unary_operation(_ti_core.expr_logic_not, lambda x: int(not x), a)


def random(dtype=float):
    """The random function.

    Args:
        dtype (DataType): Type of the random variable.

    Returns:
        A random variable whose type is `dtype`.
    """
    dtype = cook_dtype(dtype)
    x = expr.Expr(_ti_core.make_rand_expr(dtype))
    return impl.expr_init(x)


# NEXT: add matpow(self, power)


@binary
def add(a, b):
    """The add function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        sum of `a` and `b`.
    """
    return _binary_operation(_ti_core.expr_add, _bt_ops_mod.add, a, b)


@binary
def sub(a, b):
    """The sub function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        `a` subtract `b`.
    """
    return _binary_operation(_ti_core.expr_sub, _bt_ops_mod.sub, a, b)


@binary
def mul(a, b):
    """The multiply function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        `a` multiplied by `b`.
    """
    return _binary_operation(_ti_core.expr_mul, _bt_ops_mod.mul, a, b)


@binary
def mod(a, b):
    """The remainder function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix with elements not equal to zero.

    Returns:
        The remainder of `a` divided by `b`.
    """
    def expr_python_mod(a, b):
        # a % b = a - (a // b) * b
        quotient = expr.Expr(_ti_core.expr_floordiv(a, b))
        multiply = expr.Expr(_ti_core.expr_mul(b, quotient.ptr))
        return _ti_core.expr_sub(a, multiply.ptr)

    return _binary_operation(expr_python_mod, _bt_ops_mod.mod, a, b)


@binary
def pow(a, b):  # pylint: disable=W0622
    """The power function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        `a` to the `b`.
    """
    return _binary_operation(_ti_core.expr_pow, _bt_ops_mod.pow, a, b)


@binary
def floordiv(a, b):
    """The floor division function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix with elements not equal to zero.

    Returns:
        The floor function of `a` divided by `b`.
    """
    return _binary_operation(_ti_core.expr_floordiv, _bt_ops_mod.floordiv, a,
                             b)


@binary
def truediv(a, b):
    """True division function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix with elements not equal to zero.

    Returns:
        The true value of `a` divided by `b`.
    """
    return _binary_operation(_ti_core.expr_truediv, _bt_ops_mod.truediv, a, b)


@binary
def max_impl(a, b):
    """The maxnimum function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        The maxnimum of `a` and `b`.
    """
    return _binary_operation(_ti_core.expr_max, builtins.max, a, b)


@binary
def min_impl(a, b):
    """The minimum function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.

    Returns:
        The minimum of `a` and `b`.
    """
    return _binary_operation(_ti_core.expr_min, builtins.min, a, b)


@binary
def atan2(a, b):
    """The inverses of the tangent function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix with elements not equal to zero.

    Returns:
        The inverses function of tangent of `b/a`.
    """
    return _binary_operation(_ti_core.expr_atan2, math.atan2, a, b)


@binary
def raw_div(a, b):
    """Raw_div function.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix with elements not equal to zero.

    Returns:
        If `a` is a `int` and `b` is a `int`, then return `a//b`. Else return `a/b`.
    """
    def c_div(a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a // b
        return a / b

    return _binary_operation(_ti_core.expr_div, c_div, a, b)


@binary
def raw_mod(a, b):
    """Raw_mod function. Both `a` and `b` can be `float`.

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix.
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): A number or a matrix with elements not equal to zero.

    Returns:
        The remainder of `a` divided by `b`.
    """
    def c_mod(a, b):
        return a - b * int(float(a) / b)

    return _binary_operation(_ti_core.expr_mod, c_mod, a, b)


@binary
def cmp_lt(a, b):
    """Compare two values (less than)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is strictly smaller than RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_lt, lambda a, b: -int(a < b), a,
                             b)


@binary
def cmp_le(a, b):
    """Compare two values (less than or equal to)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is smaller than or equal to RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_le, lambda a, b: -int(a <= b),
                             a, b)


@binary
def cmp_gt(a, b):
    """Compare two values (greater than)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is strictly larger than RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_gt, lambda a, b: -int(a > b), a,
                             b)


@binary
def cmp_ge(a, b):
    """Compare two values (greater than or equal to)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        bool: True if LHS is greater than or equal to RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_ge, lambda a, b: -int(a >= b),
                             a, b)


@binary
def cmp_eq(a, b):
    """Compare two values (equal to)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is equal to RHS, False otherwise.

    """
    return _binary_operation(_ti_core.expr_cmp_eq, lambda a, b: -int(a == b),
                             a, b)


@binary
def cmp_ne(a, b):
    """Compare two values (not equal to)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: True if LHS is not equal to RHS, False otherwise

    """
    return _binary_operation(_ti_core.expr_cmp_ne, lambda a, b: -int(a != b),
                             a, b)


@binary
def bit_or(a, b):
    """Computes bitwise-or

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: LHS bitwise-or with RHS

    """
    return _binary_operation(_ti_core.expr_bit_or, _bt_ops_mod.or_, a, b)


@binary
def bit_and(a, b):
    """Compute bitwise-and

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: LHS bitwise-and with RHS

    """
    return _binary_operation(_ti_core.expr_bit_and, _bt_ops_mod.and_, a, b)


@binary
def bit_xor(a, b):
    """Compute bitwise-xor

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, bool]: LHS bitwise-xor with RHS

    """
    return _binary_operation(_ti_core.expr_bit_xor, _bt_ops_mod.xor, a, b)


@binary
def bit_shl(a, b):
    """Compute bitwise shift left

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, int]: LHS << RHS

    """
    return _binary_operation(_ti_core.expr_bit_shl, _bt_ops_mod.lshift, a, b)


@binary
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
@binary
def bit_shr(a, b):
    """Compute bitwise shift right (in taichi scope)

    Args:
        a (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value LHS
        b (Union[:class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`]): value RHS

    Returns:
        Union[:class:`~taichi.lang.expr.Expr`, int]: LHS >> RHS

    """
    return _binary_operation(_ti_core.expr_bit_shr, _bt_ops_mod.rshift, a, b)


# We don't have logic_and/or instructions yet:
logical_or = bit_or
logical_and = bit_and


@ternary
def select(cond, a, b):
    # TODO: systematically resolve `-1 = True` problem by introducing u1:
    cond = logical_not(logical_not(cond))

    def py_select(cond, a, b):
        return a * cond + b * (1 - cond)

    return _ternary_operation(_ti_core.expr_select, py_select, cond, a, b)


@writeback_binary
def atomic_add(a, b):
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_add(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_sub(a, b):
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_sub(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_min(a, b):
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_min(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_max(a, b):
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_max(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_and(a, b):
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_bit_and(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_or(a, b):
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_bit_or(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_xor(a, b):
    return impl.expr_init(
        expr.Expr(_ti_core.expr_atomic_bit_xor(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def assign(a, b):
    impl.get_runtime().prog.current_ast_builder().expr_assign(
        a.ptr, b.ptr, stack_info())
    return a


def max(*args):  # pylint: disable=W0622
    num_args = len(args)
    assert num_args >= 1
    if num_args == 1:
        return args[0]
    if num_args == 2:
        return max_impl(args[0], args[1])
    return max_impl(args[0], max(*args[1:]))


def min(*args):  # pylint: disable=W0622
    num_args = len(args)
    assert num_args >= 1
    if num_args == 1:
        return args[0]
    if num_args == 2:
        return min_impl(args[0], args[1])
    return min_impl(args[0], min(*args[1:]))


def ti_any(a):
    return a.any()


def ti_all(a):
    return a.all()


__all__ = [
    "acos", "asin", "atan2", "atomic_and", "atomic_or", "atomic_xor",
    "atomic_max", "atomic_sub", "atomic_min", "atomic_add", "bit_cast",
    "bit_shr", "cast", "ceil", "cos", "exp", "floor", "log", "random",
    "raw_mod", "raw_div", "round", "rsqrt", "sin", "sqrt", "tan", "tanh",
    "max", "min", "select", "abs", "pow"
]
