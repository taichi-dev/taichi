from .expr import *
from .util import *
from .impl import expr_init
import numbers
import functools

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


def unary(foo):
    import taichi as ti

    @functools.wraps(foo)
    def imp_foo(x):
        return foo(Expr(x))

    @functools.wraps(foo)
    def wrapped(a):
        if ti.is_taichi_class(a):
            return a.element_wise_unary(imp_foo)
        else:
            return foo(Expr(a))

    unary_ops.append(wrapped)
    return wrapped


binary_ops = []


def binary(foo):
    import taichi as ti

    @functools.wraps(foo)
    def imp_foo(x, y):
        return foo(Expr(x), Expr(y))

    @functools.wraps(foo)
    def rev_foo(x, y):
        return foo(Expr(y), Expr(x))

    @functools.wraps(foo)
    def wrapped(a, b):
        if ti.is_taichi_class(a):
            return a.element_wise_binary(imp_foo, b)
        elif ti.is_taichi_class(b):
            return b.element_wise_binary(rev_foo, a)
        else:
            return imp_foo(a, b)

    binary_ops.append(wrapped)
    return wrapped


writeback_binary_ops = []


def writeback_binary(foo):
    import taichi as ti

    @functools.wraps(foo)
    def imp_foo(x, y):
        return foo(x, Expr(y))

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
        return Expr(taichi_lang_core.value_cast(Expr(obj).ptr, type))


def bit_cast(obj, type):
    if is_taichi_class(obj):
        raise ValueError('Cannot apply bit_cast on Taichi classes')
    else:
        return Expr(taichi_lang_core.bits_cast(Expr(obj).ptr, type))


@deprecated('ti.sqr(x)', 'x ** 2')
def sqr(obj):
    return obj * obj


@unary
def neg(expr):
    return Expr(taichi_lang_core.expr_neg(expr.ptr), tb=stack_info())


@unary
def sin(expr):
    return Expr(taichi_lang_core.expr_sin(expr.ptr), tb=stack_info())


@unary
def cos(expr):
    return Expr(taichi_lang_core.expr_cos(expr.ptr), tb=stack_info())


@unary
def asin(expr):
    return Expr(taichi_lang_core.expr_asin(expr.ptr), tb=stack_info())


@unary
def acos(expr):
    return Expr(taichi_lang_core.expr_acos(expr.ptr), tb=stack_info())


@unary
def sqrt(expr):
    return Expr(taichi_lang_core.expr_sqrt(expr.ptr), tb=stack_info())


@unary
def floor(expr):
    return Expr(taichi_lang_core.expr_floor(expr.ptr), tb=stack_info())


@unary
def ceil(expr):
    return Expr(taichi_lang_core.expr_ceil(expr.ptr), tb=stack_info())


@unary
def inv(expr):
    return Expr(taichi_lang_core.expr_inv(expr.ptr), tb=stack_info())


@unary
def tan(expr):
    return Expr(taichi_lang_core.expr_tan(expr.ptr), tb=stack_info())


@unary
def tanh(expr):
    return Expr(taichi_lang_core.expr_tanh(expr.ptr), tb=stack_info())


@unary
def exp(expr):
    return Expr(taichi_lang_core.expr_exp(expr.ptr), tb=stack_info())


@unary
def log(expr):
    return Expr(taichi_lang_core.expr_log(expr.ptr), tb=stack_info())


@unary
def abs(expr):
    return Expr(taichi_lang_core.expr_abs(expr.ptr), tb=stack_info())


@unary
def bit_not(expr):
    return Expr(taichi_lang_core.expr_bit_not(expr.ptr), tb=stack_info())


@unary
def logical_not(expr):
    return Expr(taichi_lang_core.expr_logic_not(expr.ptr), tb=stack_info())


def random(dt=None):
    if dt is None:
        import taichi
        dt = taichi.get_runtime().default_fp
    return Expr(taichi_lang_core.make_rand_expr(dt))


@binary
def add(a, b):
    return Expr(taichi_lang_core.expr_add(a.ptr, b.ptr), tb=stack_info())


@binary
def sub(a, b):
    return Expr(taichi_lang_core.expr_sub(a.ptr, b.ptr), tb=stack_info())


@binary
def mul(a, b):
    return Expr(taichi_lang_core.expr_mul(a.ptr, b.ptr), tb=stack_info())


@binary
def mod(a, b):
    quotient = Expr(taichi_lang_core.expr_floordiv(a.ptr, b.ptr))
    multiply = Expr(taichi_lang_core.expr_mul(b.ptr, quotient.ptr))
    return Expr(taichi_lang_core.expr_sub(a.ptr, multiply.ptr))


@binary
def raw_pow(a, b):
    return Expr(taichi_lang_core.expr_pow(a.ptr, b.ptr), tb=stack_info())


# TODO: move this to a C++ pass (#944)
def pow(self, power):
    import taichi as ti
    if not isinstance(power, int):
        return raw_pow(self, power)
    if power == 0:
        # TODO: remove the hack, use {Expr,Matrix}.dup().fill(1)
        # also note that this can be solved by #940
        return self * 0 + Expr(1)

    negative = power < 0
    # Why not simply use `power = abs(power)`?
    # Because `abs` is overrided by the `ti.abs` above.
    if negative:
        power = -power

    tmp = self
    ret = None
    while power:
        if power & 1:
            if ret is None:
                ret = tmp
            else:
                ret = ti.expr_init(ret * tmp)
        tmp = ti.expr_init(tmp * tmp)
        power >>= 1

    if negative:
        return 1 / ret
    else:
        return ret


# NEXT: add matpow(self, power)


@binary
def floordiv(a, b):
    return Expr(taichi_lang_core.expr_floordiv(a.ptr, b.ptr), tb=stack_info())


@binary
def truediv(a, b):
    return Expr(taichi_lang_core.expr_truediv(a.ptr, b.ptr), tb=stack_info())


@binary
def max(a, b):
    return Expr(taichi_lang_core.expr_max(a.ptr, b.ptr), tb=stack_info())


@binary
def min(a, b):
    return Expr(taichi_lang_core.expr_min(a.ptr, b.ptr), tb=stack_info())


@binary
def atan2(a, b):
    return Expr(taichi_lang_core.expr_atan2(a.ptr, b.ptr), tb=stack_info())


@binary
def raw_div(a, b):
    return Expr(taichi_lang_core.expr_div(a.ptr, b.ptr), tb=stack_info())


@binary
def raw_mod(a, b):
    return Expr(taichi_lang_core.expr_mod(a.ptr, b.ptr), tb=stack_info())


@binary
def cmp_lt(a, b):
    return Expr(taichi_lang_core.expr_cmp_lt(a.ptr, b.ptr), tb=stack_info())


@binary
def cmp_le(a, b):
    return Expr(taichi_lang_core.expr_cmp_le(a.ptr, b.ptr), tb=stack_info())


@binary
def cmp_gt(a, b):
    return Expr(taichi_lang_core.expr_cmp_gt(a.ptr, b.ptr), tb=stack_info())


@binary
def cmp_ge(a, b):
    return Expr(taichi_lang_core.expr_cmp_ge(a.ptr, b.ptr), tb=stack_info())


@binary
def cmp_eq(a, b):
    return Expr(taichi_lang_core.expr_cmp_eq(a.ptr, b.ptr), tb=stack_info())


@binary
def cmp_ne(a, b):
    return Expr(taichi_lang_core.expr_cmp_ne(a.ptr, b.ptr), tb=stack_info())


@binary
def bit_or(a, b):
    return Expr(taichi_lang_core.expr_bit_or(a.ptr, b.ptr), tb=stack_info())


@binary
def bit_and(a, b):
    return Expr(taichi_lang_core.expr_bit_and(a.ptr, b.ptr), tb=stack_info())


@binary
def bit_xor(a, b):
    return Expr(taichi_lang_core.expr_bit_xor(a.ptr, b.ptr), tb=stack_info())


# We don't have logic_and/or instructions yet:
logical_or = bit_or
logical_and = bit_and


@writeback_binary
def atomic_add(a, b):
    return expr_init(
        Expr(taichi_lang_core.expr_atomic_add(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_sub(a, b):
    return expr_init(
        Expr(taichi_lang_core.expr_atomic_sub(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_min(a, b):
    return expr_init(
        Expr(taichi_lang_core.expr_atomic_min(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_max(a, b):
    return expr_init(
        Expr(taichi_lang_core.expr_atomic_max(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_and(a, b):
    return expr_init(
        Expr(taichi_lang_core.expr_atomic_bit_and(a.ptr, b.ptr),
             tb=stack_info()))


@writeback_binary
def atomic_or(a, b):
    return expr_init(
        Expr(taichi_lang_core.expr_atomic_bit_or(a.ptr, b.ptr),
             tb=stack_info()))


@writeback_binary
def atomic_xor(a, b):
    return expr_init(
        Expr(taichi_lang_core.expr_atomic_bit_xor(a.ptr, b.ptr),
             tb=stack_info()))


@writeback_binary
def assign(a, b):
    taichi_lang_core.expr_assign(a.ptr, b.ptr, stack_info())
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
        taichi_lang_core.insert_append(l.snode().ptr, make_expr_group(indices),
                                       Expr(val).ptr))
    return a


def is_active(l, indices):
    return Expr(
        taichi_lang_core.insert_is_active(l.snode().ptr,
                                          make_expr_group(indices)))


def deactivate(l, indices):
    taichi_lang_core.insert_deactivate(l.snode().ptr, make_expr_group(indices))


def length(l, indices):
    return Expr(
        taichi_lang_core.insert_len(l.snode().ptr, make_expr_group(indices)))
