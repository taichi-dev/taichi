from .expr import *
from .util import *
from .impl import expr_init
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
def neg(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_neg(a.ptr), tb=stack_info())
    else:
        return -a


@unary
def sin(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_sin(a.ptr), tb=stack_info())
    else:
        return math.sin(a)


@unary
def cos(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_cos(a.ptr), tb=stack_info())
    else:
        return math.cos(a)


@unary
def asin(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_asin(a.ptr), tb=stack_info())
    else:
        return math.asin(a)


@unary
def acos(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_acos(a.ptr), tb=stack_info())
    else:
        return math.acos(a)


@unary
def sqrt(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_sqrt(a.ptr), tb=stack_info())
    else:
        return math.sqrt(a)


@unary
def floor(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_floor(a.ptr), tb=stack_info())
    else:
        return math.floor(a)


@unary
def ceil(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_ceil(a.ptr), tb=stack_info())
    else:
        return math.ceil(a)


@unary
def tan(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_tan(a.ptr), tb=stack_info())
    else:
        return math.tan(a)


@unary
def tanh(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_tanh(a.ptr), tb=stack_info())
    else:
        return math.tanh(a)


@unary
def exp(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_exp(a.ptr), tb=stack_info())
    else:
        return math.exp(a)


@unary
def log(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_log(a.ptr), tb=stack_info())
    else:
        return math.log(a)


@unary
def abs(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_abs(a.ptr), tb=stack_info())
    else:
        return abs(a)


@unary
def bit_not(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_bit_not(a.ptr), tb=stack_info())
    else:
        return ~a


@unary
def logical_not(a):
    if is_taichi_expr(a):
        return Expr(taichi_lang_core.expr_logic_not(a.ptr), tb=stack_info())
    else:
        return -int(not a)


def random(dt=None):
    if dt is None:
        import taichi
        dt = taichi.get_runtime().default_fp
    return Expr(taichi_lang_core.make_rand_expr(dt))


# TODO: move this to a C++ pass (#944)
def pow(self, power):
    import taichi as ti
    if not is_taichi_expr(self) and not is_taichi_expr(power):
        # Python constant computations (#1188)
        return raw_pow(self, power)
    if not isinstance(power, int):
        return raw_pow(self, power)
    if power == 0:
        # TODO: remove the hack, use {Expr,Matrix}.dup().fill(1)
        # also note that this can be solved by #940
        return self * 0 + 1

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
def add(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_add(a.ptr, b.ptr), tb=stack_info())
    else:
        return a + b


@binary
def sub(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_sub(a.ptr, b.ptr), tb=stack_info())
    else:
        return a - b


@binary
def mul(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_mul(a.ptr, b.ptr), tb=stack_info())
    else:
        return a * b


@binary
def mod(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        # TODO: add expr_python_mod instead
        quotient = Expr(taichi_lang_core.expr_floordiv(a.ptr, b.ptr))
        multiply = Expr(taichi_lang_core.expr_mul(b.ptr, quotient.ptr))
        return Expr(taichi_lang_core.expr_sub(a.ptr, multiply.ptr))
    else:
        return a % b


@binary
def raw_pow(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_pow(a.ptr, b.ptr), tb=stack_info())
    else:
        return a**b


@binary
def floordiv(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_floordiv(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return a // b


@binary
def truediv(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_truediv(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return a / b


@binary
def max(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_max(a.ptr, b.ptr), tb=stack_info())
    else:
        import builtins
        return builtins.max(a, b)


@binary
def min(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_min(a.ptr, b.ptr), tb=stack_info())
    else:
        import builtins
        return builtins.min(a, b)


@binary
def atan2(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_atan2(a.ptr, b.ptr), tb=stack_info())
    else:
        return math.atan2(a, b)


@binary
def raw_div(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_div(a.ptr, b.ptr), tb=stack_info())
    else:
        return a // b  # TODO: Is this correct???


@binary
def raw_mod(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_mod(a.ptr, b.ptr), tb=stack_info())
    else:
        return a - b * int(float(a) / b)


@binary
def cmp_lt(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_cmp_lt(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return -int(a < b)


@binary
def cmp_le(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_cmp_le(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return -int(a <= b)


@binary
def cmp_gt(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_cmp_gt(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return -int(a >= b)


@binary
def cmp_ge(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_cmp_ge(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return -int(a > b)


@binary
def cmp_eq(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_cmp_eq(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return -int(a == b)


@binary
def cmp_ne(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_cmp_ne(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return -int(a != b)


@binary
def bit_or(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_bit_or(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return a | b


@binary
def bit_and(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_bit_and(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return a & b


@binary
def bit_xor(a, b):
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_lang_core.expr_bit_xor(a.ptr, b.ptr),
                    tb=stack_info())
    else:
        return a ^ b


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
