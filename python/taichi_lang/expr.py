from .core import taichi_lang_core
from .util import is_taichi_class
import sys
import traceback

# Scalar, basic data type
class Expr:
  materialize_layout_callback = None
  layout_materialized = False

  def __init__(self, *args, tb=None):
    self.getter = None
    self.setter = None
    self.tb = tb
    if len(args) == 1:
      if isinstance(args[0], taichi_lang_core.Expr):
        self.ptr = args[0]
      elif isinstance(args[0], Expr):
        self.ptr = args[0].ptr
        self.tb = args[0].tb
      else:
        self.ptr = taichi_lang_core.make_constant_expr(args[0])
    else:
      assert False
    if self.tb:
      self.ptr.set_tb(self.tb)

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


  def __sub__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_sub(self.ptr, other.ptr))

  def __isub__(self, other):
    self.assign(Expr(taichi_lang_core.expr_sub(self.ptr, other.ptr)))

  def __imul__(self, other):
    self.assign(Expr(taichi_lang_core.expr_mul(self.ptr, other.ptr)))

  def __idiv__(self, other):
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

  def assign(self, other):
    taichi_lang_core.expr_assign(self.ptr, Expr(other).ptr)

  def serialize(self):
    return self.ptr.serialize()

  def initialize_accessor(self):
    if self.getter:
      return
    snode = self.ptr.snode()
    num_ind = snode.num_active_indices()
    dt_name = taichi_lang_core.data_type_short_name(snode.data_type())
    self.getter = getattr(self.ptr, 'val{}_{}'.format(num_ind, dt_name))
    self.setter = getattr(self.ptr, 'set_val{}_{}'.format(num_ind, dt_name))

  def __setitem__(self, key, value):
    if not Expr.layout_materialized:
      self.materialize_layout_callback()
    self.initialize_accessor()
    if not isinstance(key, tuple):
      key = (key, )
    self.setter(value, *key)

  def __getitem__(self, key):
    if not Expr.layout_materialized:
      self.materialize_layout_callback()
    self.initialize_accessor()
    if not isinstance(key, tuple):
      key = (key, )
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
    else:
      assert False, op

  def set_grad(self, grad):
    self.ptr.set_grad(grad.ptr)
