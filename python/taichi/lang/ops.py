import builtins
import ctypes
import functools
import math
import operator as _bt_ops_mod  # bt for builtin
import traceback

from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl, matrix
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.expr import Expr, make_expr_group
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
    return isinstance(a, Expr)


def wrap_if_not_expr(a):
    _taichi_skip_traceback = 1
    return Expr(a) if not is_taichi_expr(a) else a


def unary(foo):
    @functools.wraps(foo)
    def imp_foo(x):
        _taichi_skip_traceback = 2
        return foo(x)

    @functools.wraps(foo)
    def wrapped(a):
        _taichi_skip_traceback = 1
        if is_taichi_class(a):
            return a.element_wise_unary(imp_foo)
        else:
            return imp_foo(a)

    return wrapped


binary_ops = []


def binary(foo):
    @functools.wraps(foo)
    def imp_foo(x, y):
        _taichi_skip_traceback = 2
        return foo(x, y)

    @functools.wraps(foo)
    def rev_foo(x, y):
        _taichi_skip_traceback = 2
        return foo(y, x)

    @functools.wraps(foo)
    def wrapped(a, b):
        _taichi_skip_traceback = 1
        if is_taichi_class(a):
            return a.element_wise_binary(imp_foo, b)
        elif is_taichi_class(b):
            return b.element_wise_binary(rev_foo, a)
        else:
            return imp_foo(a, b)

    binary_ops.append(wrapped)
    return wrapped


ternary_ops = []


def ternary(foo):
    @functools.wraps(foo)
    def abc_foo(a, b, c):
        _taichi_skip_traceback = 2
        return foo(a, b, c)

    @functools.wraps(foo)
    def bac_foo(b, a, c):
        _taichi_skip_traceback = 2
        return foo(a, b, c)

    @functools.wraps(foo)
    def cab_foo(c, a, b):
        _taichi_skip_traceback = 2
        return foo(a, b, c)

    @functools.wraps(foo)
    def wrapped(a, b, c):
        _taichi_skip_traceback = 1
        if is_taichi_class(a):
            return a.element_wise_ternary(abc_foo, b, c)
        elif is_taichi_class(b):
            return b.element_wise_ternary(bac_foo, a, c)
        elif is_taichi_class(c):
            return c.element_wise_ternary(cab_foo, a, b)
        else:
            return abc_foo(a, b, c)

    ternary_ops.append(wrapped)
    return wrapped


writeback_binary_ops = []


def writeback_binary(foo):
    @functools.wraps(foo)
    def imp_foo(x, y):
        _taichi_skip_traceback = 2
        return foo(x, wrap_if_not_expr(y))

    @functools.wraps(foo)
    def wrapped(a, b):
        _taichi_skip_traceback = 1
        if is_taichi_class(a):
            return a.element_wise_writeback_binary(imp_foo, b)
        elif is_taichi_class(b):
            raise TaichiSyntaxError(
                f'cannot augassign taichi class {type(b)} to scalar expr')
        else:
            return imp_foo(a, b)

    writeback_binary_ops.append(wrapped)
    return wrapped


def cast(obj, dtype):
    _taichi_skip_traceback = 1
    dtype = cook_dtype(dtype)
    if is_taichi_class(obj):
        # TODO: unify with element_wise_unary
        return obj.cast(dtype)
    else:
        return Expr(_ti_core.value_cast(Expr(obj).ptr, dtype))


def bit_cast(obj, dtype):
    _taichi_skip_traceback = 1
    dtype = cook_dtype(dtype)
    if is_taichi_class(obj):
        raise ValueError('Cannot apply bit_cast on Taichi classes')
    else:
        return Expr(_ti_core.bits_cast(Expr(obj).ptr, dtype))


def _unary_operation(taichi_op, python_op, a):
    _taichi_skip_traceback = 1
    if is_taichi_expr(a):
        return Expr(taichi_op(a.ptr), tb=stack_info())
    else:
        return python_op(a)


def _binary_operation(taichi_op, python_op, a, b):
    _taichi_skip_traceback = 1
    if is_taichi_expr(a) or is_taichi_expr(b):
        a, b = wrap_if_not_expr(a), wrap_if_not_expr(b)
        return Expr(taichi_op(a.ptr, b.ptr), tb=stack_info())
    else:
        return python_op(a, b)


def _ternary_operation(taichi_op, python_op, a, b, c):
    _taichi_skip_traceback = 1
    if is_taichi_expr(a) or is_taichi_expr(b) or is_taichi_expr(c):
        a, b, c = wrap_if_not_expr(a), wrap_if_not_expr(b), wrap_if_not_expr(c)
        return Expr(taichi_op(a.ptr, b.ptr, c.ptr), tb=stack_info())
    else:
        return python_op(a, b, c)


@unary
def neg(a):
    return _unary_operation(_ti_core.expr_neg, _bt_ops_mod.neg, a)


@unary
def sin(a):
    return _unary_operation(_ti_core.expr_sin, math.sin, a)


@unary
def cos(a):
    return _unary_operation(_ti_core.expr_cos, math.cos, a)


@unary
def asin(a):
    return _unary_operation(_ti_core.expr_asin, math.asin, a)


@unary
def acos(a):
    return _unary_operation(_ti_core.expr_acos, math.acos, a)


@unary
def sqrt(a):
    return _unary_operation(_ti_core.expr_sqrt, math.sqrt, a)


@unary
def rsqrt(a):
    def _rsqrt(a):
        return 1 / math.sqrt(a)

    return _unary_operation(_ti_core.expr_rsqrt, _rsqrt, a)


@unary
def floor(a):
    return _unary_operation(_ti_core.expr_floor, math.floor, a)


@unary
def ceil(a):
    return _unary_operation(_ti_core.expr_ceil, math.ceil, a)


@unary
def tan(a):
    return _unary_operation(_ti_core.expr_tan, math.tan, a)


@unary
def tanh(a):
    return _unary_operation(_ti_core.expr_tanh, math.tanh, a)


@unary
def exp(a):
    return _unary_operation(_ti_core.expr_exp, math.exp, a)


@unary
def log(a):
    return _unary_operation(_ti_core.expr_log, math.log, a)


@unary
def abs(a):
    return _unary_operation(_ti_core.expr_abs, builtins.abs, a)


@unary
def bit_not(a):
    return _unary_operation(_ti_core.expr_bit_not, _bt_ops_mod.invert, a)


@unary
def logical_not(a):
    return _unary_operation(_ti_core.expr_logic_not, lambda x: int(not x), a)


def random(dtype=float):
    dtype = cook_dtype(dtype)
    x = Expr(_ti_core.make_rand_expr(dtype))
    return impl.expr_init(x)


# NEXT: add matpow(self, power)


@binary
def add(a, b):
    return _binary_operation(_ti_core.expr_add, _bt_ops_mod.add, a, b)


@binary
def sub(a, b):
    return _binary_operation(_ti_core.expr_sub, _bt_ops_mod.sub, a, b)


@binary
def mul(a, b):
    return _binary_operation(_ti_core.expr_mul, _bt_ops_mod.mul, a, b)


@binary
def mod(a, b):
    def expr_python_mod(a, b):
        # a % b = (a // b) * b - a
        quotient = Expr(_ti_core.expr_floordiv(a, b))
        multiply = Expr(_ti_core.expr_mul(b, quotient.ptr))
        return _ti_core.expr_sub(a, multiply.ptr)

    return _binary_operation(expr_python_mod, _bt_ops_mod.mod, a, b)


@binary
def pow(a, b):
    return _binary_operation(_ti_core.expr_pow, _bt_ops_mod.pow, a, b)


@binary
def floordiv(a, b):
    return _binary_operation(_ti_core.expr_floordiv, _bt_ops_mod.floordiv, a,
                             b)


@binary
def truediv(a, b):
    return _binary_operation(_ti_core.expr_truediv, _bt_ops_mod.truediv, a, b)


@binary
def max(a, b):
    return _binary_operation(_ti_core.expr_max, builtins.max, a, b)


@binary
def min(a, b):
    return _binary_operation(_ti_core.expr_min, builtins.min, a, b)


@binary
def atan2(a, b):
    return _binary_operation(_ti_core.expr_atan2, math.atan2, a, b)


@binary
def raw_div(a, b):
    def c_div(a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a // b
        else:
            return a / b

    return _binary_operation(_ti_core.expr_div, c_div, a, b)


@binary
def raw_mod(a, b):
    def c_mod(a, b):
        return a - b * int(float(a) / b)

    return _binary_operation(_ti_core.expr_mod, c_mod, a, b)


@binary
def cmp_lt(a, b):
    return _binary_operation(_ti_core.expr_cmp_lt, lambda a, b: -int(a < b), a,
                             b)


@binary
def cmp_le(a, b):
    return _binary_operation(_ti_core.expr_cmp_le, lambda a, b: -int(a <= b),
                             a, b)


@binary
def cmp_gt(a, b):
    return _binary_operation(_ti_core.expr_cmp_gt, lambda a, b: -int(a > b), a,
                             b)


@binary
def cmp_ge(a, b):
    return _binary_operation(_ti_core.expr_cmp_ge, lambda a, b: -int(a >= b),
                             a, b)


@binary
def cmp_eq(a, b):
    return _binary_operation(_ti_core.expr_cmp_eq, lambda a, b: -int(a == b),
                             a, b)


@binary
def cmp_ne(a, b):
    return _binary_operation(_ti_core.expr_cmp_ne, lambda a, b: -int(a != b),
                             a, b)


@binary
def bit_or(a, b):
    return _binary_operation(_ti_core.expr_bit_or, _bt_ops_mod.or_, a, b)


@binary
def bit_and(a, b):
    return _binary_operation(_ti_core.expr_bit_and, _bt_ops_mod.and_, a, b)


@binary
def bit_xor(a, b):
    return _binary_operation(_ti_core.expr_bit_xor, _bt_ops_mod.xor, a, b)


@binary
def bit_shl(a, b):
    return _binary_operation(_ti_core.expr_bit_shl, _bt_ops_mod.lshift, a, b)


@binary
def bit_sar(a, b):
    return _binary_operation(_ti_core.expr_bit_sar, _bt_ops_mod.rshift, a, b)


@taichi_scope
@binary
def bit_shr(a, b):
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
        Expr(_ti_core.expr_atomic_add(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_sub(a, b):
    return impl.expr_init(
        Expr(_ti_core.expr_atomic_sub(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_min(a, b):
    return impl.expr_init(
        Expr(_ti_core.expr_atomic_min(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_max(a, b):
    return impl.expr_init(
        Expr(_ti_core.expr_atomic_max(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_and(a, b):
    return impl.expr_init(
        Expr(_ti_core.expr_atomic_bit_and(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_or(a, b):
    return impl.expr_init(
        Expr(_ti_core.expr_atomic_bit_or(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def atomic_xor(a, b):
    return impl.expr_init(
        Expr(_ti_core.expr_atomic_bit_xor(a.ptr, b.ptr), tb=stack_info()))


@writeback_binary
def assign(a, b):
    _ti_core.expr_assign(a.ptr, b.ptr, stack_info())
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
    return a.any()


def ti_all(a):
    return a.all()


def append(l, indices, val):
    a = impl.expr_init(
        _ti_core.insert_append(l.snode.ptr, make_expr_group(indices),
                               Expr(val).ptr))
    return a


def external_func_call(func, args=[], outputs=[]):
    func_addr = ctypes.cast(func, ctypes.c_void_p).value
    _ti_core.insert_external_func_call(func_addr, '', make_expr_group(args),
                                       make_expr_group(outputs))


def asm(source, inputs=[], outputs=[]):

    _ti_core.insert_external_func_call(0, source, make_expr_group(inputs),
                                       make_expr_group(outputs))


def is_active(l, indices):
    return Expr(
        _ti_core.insert_is_active(l.snode.ptr, make_expr_group(indices)))


def activate(l, indices):
    _ti_core.insert_activate(l.snode.ptr, make_expr_group(indices))


def deactivate(l, indices):
    _ti_core.insert_deactivate(l.snode.ptr, make_expr_group(indices))


def length(l, indices):
    return Expr(_ti_core.insert_len(l.snode.ptr, make_expr_group(indices)))


def rescale_index(a, b, I):
    """Rescales the index 'I' of field 'a' the match the shape of field 'b'

    Parameters
    ----------
    a: ti.field(), ti.Vector.field, ti.Matrix.field()
        input taichi field
    b: ti.field(), ti.Vector.field, ti.Matrix.field()
        output taichi field
    I: ti.Vector()
        grouped loop index

    Returns
    -------
    Ib: ti.Vector()
        rescaled grouped loop index

    """
    assert isinstance(a, Expr) and a.is_global(), \
            f"first arguement must be a field"
    assert isinstance(b, Expr) and b.is_global(), \
            f"second arguement must be a field"
    assert isinstance(I, matrix.Matrix) and not I.is_global(), \
            f"third arguement must be a grouped index"
    Ib = I.copy()
    for n in range(min(I.n, min(len(a.shape), len(b.shape)))):
        if a.shape[n] > b.shape[n]:
            Ib.entries[n] = I.entries[n] // (a.shape[n] // b.shape[n])
        if a.shape[n] < b.shape[n]:
            Ib.entries[n] = I.entries[n] * (b.shape[n] // a.shape[n])
    return Ib


def get_addr(f, indices):
    """Returns the address (for CUDA/x64) for index `indices` of field `f`. Currently, this function can only be called inside a kernel.

    Args:
        f (ti.field(), ti.Vector.field, ti.Matrix.field()): input taichi field
        indices (int, ti.Vector()): field indices

    Returns:
        ti.u64:  The address of `f[indices]`.

    """
    return Expr(_ti_core.expr_get_addr(f.snode.ptr, make_expr_group(indices)))
