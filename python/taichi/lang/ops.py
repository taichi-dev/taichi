from .expr import *
from .util import *
from .impl import expr_init
from .util import taichi_lang_core as ti_core
import operator as ops
import numbers
import functools
import math

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
    return isinstance(a, Expr)


def wrap_if_not_expr(a):
    return Expr(a) if not is_taichi_expr(a) else a


def unary(foo):
    import taichi as ti

    @functools.wraps(foo)
    def wrapped(a):
        if ti.is_taichi_class(a):
            return a.element_wise_unary(foo)
        else:
            return foo(a)

    unary_ops.append(wrapped)
    return wrapped


binary_ops = []


def binary(foo):
    import taichi as ti

    @functools.wraps(foo)
    def rev_foo(x, y):
        return foo(y, x)

    @functools.wraps(foo)
    def wrapped(a, b):
        if ti.is_taichi_class(a):
            return a.element_wise_binary(foo, b)
        elif ti.is_taichi_class(b):
            return b.element_wise_binary(rev_foo, a)
        else:
            return foo(a, b)

    binary_ops.append(wrapped)
    return wrapped


writeback_binary_ops = []


def writeback_binary(foo):
    import taichi as ti

    @functools.wraps(foo)
    def imp_foo(x, y):
        return foo(x, wrap_if_not_expr(y))

    @functools.wraps(foo)
    def wrapped(a, b):
        if ti.is_taichi_class(a):
            return a.element_wise_binary(imp_foo, b)
        elif ti.is_taichi_class(b):
            raise SyntaxError(
                f'cannot augassign taichi class {type(b)} to scalar expr')
        else:
            return imp_foo(a, b)

    writeback_binary_ops.append(wrapped)
    return wrapped


def cast(obj, type):
    if is_taichi_class(obj):
        return obj.cast(type)
    else:
        return Expr(ti_core.value_cast(Expr(obj).ptr, type))


def bit_cast(obj, type):
    if is_taichi_class(obj):
        raise ValueError('Cannot apply bit_cast on Taichi classes')
    else:
        return Expr(ti_core.bits_cast(Expr(obj).ptr, type))


@deprecated('ti.sqr(x)', 'x ** 2')
def sqr(obj):
    return obj * obj


def _unary_operation(taichi_op, python_op, a):
    if is_taichi_expr(a):
        return Expr(taichi_op(a.ptr), tb=stack_info())
    else:
        return python_op(a)


@unary
def neg(a):
    return _unary_operation(ti_core.expr_neg, ops.neg, a)


@unary
def sin(a):
    return _unary_operation(ti_core.expr_sin, math.sin, a)


@unary
def cos(a):
    return _unary_operation(ti_core.expr_cos, math.cos, a)


@unary
def asin(a):
    return _unary_operation(ti_core.expr_asin, math.asin, a)


@unary
def acos(a):
    return _unary_operation(ti_core.expr_acos, math.acos, a)


@unary
def sqrt(a):
    return _unary_operation(ti_core.expr_sqrt, math.sqrt, a)


@unary
def rsqrt(a):
    def _rsqrt(a):
        return 1 / math.sqrt(a)

    return _unary_operation(ti_core.expr_rsqrt, _rsqrt, a)


@unary
def floor(a):
    return _unary_operation(ti_core.expr_floor, math.floor, a)


@unary
def ceil(a):
    return _unary_operation(ti_core.expr_ceil, math.ceil, a)


@unary
def tan(a):
    return _unary_operation(ti_core.expr_tan, math.tan, a)


@unary
def tanh(a):
    return _unary_operation(ti_core.expr_tanh, math.tanh, a)


@unary
def exp(a):
    return _unary_operation(ti_core.expr_exp, math.exp, a)


@unary
def log(a):
    return _unary_operation(ti_core.expr_log, math.log, a)


@unary
def abs(a):
    import builtins
    return _unary_operation(ti_core.expr_abs, builtins.abs, a)


@unary
def bit_not(a):
    return _unary_operation(ti_core.expr_bit_not, ops.invert, a)


@unary
def logical_not(a):
    return _unary_operation(ti_core.expr_logic_not, lambda x: int(not x), a)


def random(dt=None):
    if dt is None:
        import taichi
        dt = taichi.get_runtime().default_fp
    return Expr(ti_core.make_rand_expr(dt))


# NEXT: add matpow(self, power)


def _binary_operation(taichi_op, python_op, a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_op(a.ptr, b.ptr), tb=stack_info())
    else:
        return python_op(a, b)


@binary
def add(a, b):
    return _binary_operation(ti_core.expr_add, ops.add, a, b)


@binary
def sub(a, b):
    return _binary_operation(ti_core.expr_sub, ops.sub, a, b)


@binary
def mul(a, b):
    return _binary_operation(ti_core.expr_mul, ops.mul, a, b)


@binary
def mod(a, b):
    def expr_python_mod(a, b):
        quotient = Expr(ti_core.expr_floordiv(a, b))
        multiply = Expr(ti_core.expr_mul(b, quotient.ptr))
        return ti_core.expr_sub(a, multiply.ptr)

    return _binary_operation(expr_python_mod, ops.mod, a, b)


@binary
def pow(a, b):
    return _binary_operation(ti_core.expr_pow, ops.pow, a, b)


@binary
def floordiv(a, b):
    return _binary_operation(ti_core.expr_floordiv, ops.floordiv, a, b)


@binary
def truediv(a, b):
    return _binary_operation(ti_core.expr_truediv, ops.truediv, a, b)


@binary
def max(a, b):
    import builtins
    return _binary_operation(ti_core.expr_max, builtins.max, a, b)


@binary
def min(a, b):
    import builtins
    return _binary_operation(ti_core.expr_min, builtins.min, a, b)


@binary
def atan2(a, b):
    return _binary_operation(ti_core.expr_atan2, math.atan2, a, b)


@binary
def raw_div(a, b):
    def c_div(a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a // b
        else:
            return a / b

    return _binary_operation(ti_core.expr_div, c_div, a, b)


@binary
def raw_mod(a, b):
    def c_mod(a, b):
        return a - b * int(float(a) / b)

    return _binary_operation(ti_core.expr_mod, c_mod, a, b)


@binary
def cmp_lt(a, b):
    return _binary_operation(ti_core.expr_cmp_lt, lambda a, b: -int(a < b), a,
                             b)


@binary
def cmp_le(a, b):
    return _binary_operation(ti_core.expr_cmp_le, lambda a, b: -int(a <= b), a,
                             b)


@binary
def cmp_gt(a, b):
    return _binary_operation(ti_core.expr_cmp_gt, lambda a, b: -int(a > b), a,
                             b)


@binary
def cmp_ge(a, b):
    return _binary_operation(ti_core.expr_cmp_ge, lambda a, b: -int(a >= b), a,
                             b)


@binary
def cmp_eq(a, b):
    return _binary_operation(ti_core.expr_cmp_eq, lambda a, b: -int(a == b), a,
                             b)


@binary
def cmp_ne(a, b):
    return _binary_operation(ti_core.expr_cmp_ne, lambda a, b: -int(a != b), a,
                             b)


@binary
def bit_or(a, b):
    return _binary_operation(ti_core.expr_bit_or, ops.or_, a, b)


@binary
def bit_and(a, b):
    return _binary_operation(ti_core.expr_bit_and, ops.and_, a, b)


@binary
def bit_xor(a, b):
    return _binary_operation(ti_core.expr_bit_xor, ops.xor, a, b)


# We don't have logic_and/or instructions yet:
logical_or = bit_or
logical_and = bit_and


@writeback_binary
def atomic_add(a, b):
    return expr_init(
        Expr(ti_core.expr_atomic_add(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_sub(a, b):
    return expr_init(
        Expr(ti_core.expr_atomic_sub(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_min(a, b):
    return expr_init(
        Expr(ti_core.expr_atomic_min(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_max(a, b):
    return expr_init(
        Expr(ti_core.expr_atomic_max(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_and(a, b):
    return expr_init(
        Expr(ti_core.expr_atomic_bit_and(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_or(a, b):
    return expr_init(
        Expr(ti_core.expr_atomic_bit_or(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_xor(a, b):
    return expr_init(
        Expr(ti_core.expr_atomic_bit_xor(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def assign(a, b):
    ti_core.expr_assign(a.ptr, b.ptr, stack_info())
    return a


def ti_max(*args):
    num_args = len(args)
    assert num_args >= 1
    if num_args == 1:
        return args[0]
    elif num_args == 2:
        return max(args[0], args[1])
    else:
        return max(args[0], ti_max(*args[1:]))


def ti_min(*args):
    num_args = len(args)
    assert num_args >= 1
    if num_args == 1:
        return args[0]
    elif num_args == 2:
        return min(args[0], args[1])
    else:
        return min(args[0], ti_min(*args[1:]))


def ti_any(a):
    assert hasattr(a, 'any')
    return a.any()


def ti_all(a):
    assert hasattr(a, 'all')
    return a.all()


def append(l, indices, val):
    import taichi as ti
    a = ti.expr_init(
        ti_core.insert_append(l.snode().ptr, make_expr_group(indices),
                              Expr(val).ptr))
    return a


def is_active(l, indices):
    return Expr(
        ti_core.insert_is_active(l.snode().ptr, make_expr_group(indices)))


def activate(l, indices):
    ti_core.insert_activate(l.snode().ptr, make_expr_group(indices))


def deactivate(l, indices):
    ti_core.insert_deactivate(l.snode().ptr, make_expr_group(indices))


def length(l, indices):
    return Expr(ti_core.insert_len(l.snode().ptr, make_expr_group(indices)))
