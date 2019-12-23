from .expr import *
from .util import *
import numbers

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


def unary(x):

  def func(expr):
    return x(Expr(expr))

  unary_ops.append(func)
  return func


binary_ops = []


def binary(foo):

  def x_(a, b):
    return foo(Expr(a), Expr(b))

  binary_ops.append(x_)
  return x_


def pow(x, n):
  assert isinstance(n, int) and n >= 0
  if n == 0:
    return 1
  ret = x
  for i in range(n - 1):
    ret = ret * x
  return ret


def logical_and(a, b):
  return a.logical_and(b)


def logical_or(a, b):
  return a.logical_or(b)


def logical_not(a):
  return a.logical_not()


def cast(obj, type):
  if is_taichi_class(obj):
    return obj.cast(type)
  else:
    return Expr(taichi_lang_core.value_cast(Expr(obj).ptr, type))


def sqr(obj):
  return obj * obj


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
  return Expr(taichi_lang_core.expr_inv(expr.ptr))


@unary
def tan(expr):
  return Expr(taichi_lang_core.expr_tan(expr.ptr))


@unary
def tanh(expr):
  return Expr(taichi_lang_core.expr_tanh(expr.ptr))


@unary
def exp(expr):
  return Expr(taichi_lang_core.expr_exp(expr.ptr))


@unary
def log(expr):
  return Expr(taichi_lang_core.expr_log(expr.ptr))


@unary
def abs(expr):
  return Expr(taichi_lang_core.expr_abs(expr.ptr))


def random(dt=None):
  if dt is None:
    import taichi
    dt = taichi.get_runtime().default_fp
  return Expr(taichi_lang_core.make_rand_expr(dt))


@binary
def max(a, b):
  return Expr(taichi_lang_core.expr_max(a.ptr, b.ptr))


@binary
def min(a, b):
  return Expr(taichi_lang_core.expr_min(a.ptr, b.ptr))


def ti_max(*args):
  num_args = len(args)
  assert num_args >= 1
  if num_args == 1:
    return args[0]
  elif num_args == 2:
    if isinstance(args[0], numbers.Number) and isinstance(
        args[1], numbers.Number):
      return max(args[0], args[1])
    else:
      return Expr(
          taichi_lang_core.expr_max(Expr(args[0]).ptr,
                                    Expr(args[1]).ptr))
  else:
    return ti_max(args[0], ti_max(*args[1:]))


def ti_min(*args):
  num_args = len(args)
  assert num_args >= 1
  if num_args == 1:
    return args[0]
  elif num_args == 2:
    if isinstance(args[0], numbers.Number) and isinstance(
        args[1], numbers.Number):
      return min(args[0], args[1])
    else:
      return Expr(
          taichi_lang_core.expr_min(Expr(args[0]).ptr,
                                    Expr(args[1]).ptr))
  else:
    return ti_min(args[0], ti_min(*args[1:]))


def append(l, indices, val):
  taichi_lang_core.insert_append(l.ptr, make_expr_group(indices), Expr(val).ptr)


def length(l, indices):
  return taichi_lang_core.insert_len(l.ptr, make_expr_group(indices))
