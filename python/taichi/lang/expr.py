from .core import taichi_lang_core
from .util import *
import traceback


# Scalar, basic data type
class Expr:
  materialize_layout_callback = None
  layout_materialized = False

  def __init__(self, *args, tb=None):
    self.getter = None
    self.setter = None
    self.tb = tb
    self.from_torch_ = None
    self.to_torch_ = None
    if len(args) == 1:
      if isinstance(args[0], taichi_lang_core.Expr):
        self.ptr = args[0]
      elif isinstance(args[0], Expr):
        self.ptr = args[0].ptr
        self.tb = args[0].tb
      else:
        arg = args[0]
        try:
          import numpy as np
          if isinstance(arg, np.ndarray):
            arg = arg.dtype(arg)
        except:
          pass
        from .impl import make_constant_expr
        self.ptr = make_constant_expr(arg).ptr
    else:
      assert False
    if self.tb:
      self.ptr.set_tb(self.tb)
    self.grad = None
    self.val = self

  @staticmethod
  def stack_info():
    s = traceback.extract_stack()[3:-1]
    raw = ''.join(traceback.format_list(s))
    # remove the confusing last line
    return '\n'.join(raw.split('\n')[:-2]) + '\n'

  def __add__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_add(self.ptr, other.ptr), tb=self.stack_info())
  __radd__ = __add__

  def __iadd__(self, other):
    self.assign(Expr(taichi_lang_core.expr_add(self.ptr, other.ptr)))


  def __neg__(self):
    return Expr(taichi_lang_core.expr_neg(self.ptr), tb=self.stack_info())

  def __sub__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_sub(self.ptr, other.ptr), tb=self.stack_info())

  def __isub__(self, other):
    self.assign(Expr(taichi_lang_core.expr_sub(self.ptr, other.ptr)))

  def __imul__(self, other):
    self.assign(Expr(taichi_lang_core.expr_mul(self.ptr, other.ptr)))

  def __idiv__(self, other):
    self.assign(Expr(taichi_lang_core.expr_div(self.ptr, other.ptr)))

  __itruediv__ = __idiv__

  def __ifloordiv__(self, other):
    self.assign(Expr(taichi_lang_core.expr_div(self.ptr, other.ptr)))

  def __rsub__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_sub(other.ptr, self.ptr))

  def __isub__(self, other):
    self.assign(Expr(taichi_lang_core.expr_sub(self.ptr, other.ptr)))

  def __mul__(self, other):
    if is_taichi_class(other) and hasattr(other, '__rmul__'):
      return other.__rmul__(self)
    else:
      other = Expr(other)
      return Expr(taichi_lang_core.expr_mul(self.ptr, other.ptr))

  __rmul__ = __mul__

  def __mod__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_mod(self.ptr, other.ptr))

  def __truediv__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_div(self.ptr, other.ptr))

  def __rtruediv__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_div(other.ptr, self.ptr))

  __floordiv__ = __truediv__
  __rfloordiv__ = __rtruediv__

  def __le__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_le(self.ptr, other.ptr))

  def __lt__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_lt(self.ptr, other.ptr))

  def __ge__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_ge(self.ptr, other.ptr))

  def __gt__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_gt(self.ptr, other.ptr))

  def __eq__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_eq(self.ptr, other.ptr))

  def __ne__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_ne(self.ptr, other.ptr))

  def __getitem__(self, item):
    item = Expr(item)
    return Expr(expr_index(self, item.ptr))

  def __and__(self, item):
    item = Expr(item)
    return Expr(taichi_lang_core.expr_bit_and(self.ptr, item.ptr))

  def __or__(self, item):
    item = Expr(item)
    return Expr(taichi_lang_core.expr_bit_or(self.ptr, item.ptr))


  def logical_and(self, item):
    return self & item

  def logical_or(self, item):
    return self | item

  def logical_not(self):
    return Expr(taichi_lang_core.expr_bit_not(self.ptr), tb=self.stack_info())

  def assign(self, other):
    taichi_lang_core.expr_assign(self.ptr, Expr(other).ptr, self.stack_info())

  def serialize(self):
    return self.ptr.serialize()

  def initialize_accessor(self):
    if self.getter:
      return
    snode = self.ptr.snode()
    
    if self.snode().data_type() == f32 or self.snode().data_type() == f64:
      def getter(*key):
        return snode.read_float(key[0], key[1], key[2], key[3])
      def setter(value, *key):
        snode.write_float(key[0], key[1], key[2], key[3], value)
    else:
      def getter(*key):
        return snode.read_int(key[0], key[1], key[2], key[3])
      def setter(value, *key):
        snode.write_int(key[0], key[1], key[2], key[3], value)
        
    self.getter = getter
    self.setter = setter

  def __setitem__(self, key, value):
    if not Expr.layout_materialized:
      self.materialize_layout_callback()
    self.initialize_accessor()
    if key is None:
      key = ()
    if not isinstance(key, tuple):
      key = (key, )
    key = key + ((0, ) * (4 - len(key)))
    self.setter(value, *key)

  def __getitem__(self, key):
    if not Expr.layout_materialized:
      self.materialize_layout_callback()
    self.initialize_accessor()
    if key is None:
      key = ()
    if not isinstance(key, tuple):
      key = (key, )
    key = key + ((0, ) * (4 - len(key)))
    return self.getter(*key)

  def loop_range(self):
    return self

  def augassign(self, x, op):
    x = Expr(x)
    if op == 'Add':
      self += x
    elif op == 'Sub':
      self -= x
    elif op == 'Mult':
      self *= x
    elif op == 'Div':
      self /= x
    elif op == 'FloorDiv':
      self //= x
    else:
      assert False, op

  def set_grad(self, grad):
    self.grad = grad
    self.ptr.set_grad(grad.ptr)

  def clear(self, deactivate=False):
    assert not deactivate
    node = self.ptr.snode().parent
    assert node
    node.clear_data()

  def atomic_add(self, other):
    taichi_lang_core.expr_atomic_add(self.ptr, other.ptr)

  def __pow__(self, power, modulo=None):
    assert isinstance(power, int) and power >= 0
    if power == 0:
      return Expr(1)
    ret = self
    for i in range(power - 1):
      ret = ret * self
    return ret

  def __abs__(self):
    import taichi as ti
    return ti.abs(self)

  def __ti_int__(self):
    import taichi as ti
    return ti.cast(self, ti.get_runtime().default_ip)

  def __ti_float__(self):
    import taichi as ti
    return ti.cast(self, ti.get_runtime().default_fp)

  def parent(self):
    from .snode import SNode
    return SNode(self.ptr.snode().parent)
  
  def snode(self):
    from .snode import SNode
    return SNode(self.ptr.snode())
  
  def __hash__(self):
    return self.ptr.get_raw_address()

def make_expr_group(*exprs):
  if len(exprs) == 1:
    from .matrix import Matrix
    if (isinstance(exprs[0], list) or isinstance(exprs[0], tuple)):
      exprs = exprs[0]
    elif isinstance(exprs[0], Matrix):
      mat = exprs[0]
      assert mat.m == 1
      exprs = mat.entries
  expr_group = taichi_lang_core.ExprGroup()
  for i in exprs:
    expr_group.push_back(Expr(i).ptr)
  return expr_group

