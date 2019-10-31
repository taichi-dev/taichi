from .expr import *
from .util import *

unary_ops = []


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
  return Expr(taichi_lang_core.expr_sin(expr.ptr))


@unary
def cos(expr):
  return Expr(taichi_lang_core.expr_cos(expr.ptr))

@unary
def asin(expr):
  return Expr(taichi_lang_core.expr_asin(expr.ptr))

@unary
def acos(expr):
  return Expr(taichi_lang_core.expr_acos(expr.ptr))

@unary
def sqrt(expr):
  return Expr(taichi_lang_core.expr_sqrt(expr.ptr))


@unary
def floor(expr):
  return Expr(taichi_lang_core.expr_floor(expr.ptr))


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


def random(dt=f32):
  return Expr(taichi_lang_core.make_rand_expr(dt))


@binary
def max(a, b):
  return Expr(taichi_lang_core.expr_max(a.ptr, b.ptr))


@binary
def min(a, b):
  return Expr(taichi_lang_core.expr_min(a.ptr, b.ptr))


def append(l, indices, val):
  taichi_lang_core.insert_append(l.ptr, make_expr_group(indices), Expr(val).ptr)


def length(l, indices):
  return taichi_lang_core.insert_len(l.ptr, make_expr_group(indices))
