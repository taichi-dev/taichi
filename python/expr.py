from core import taichi_lang


class Expr:
  def __init__(self, *args):
    if len(args) == 1:
      if isinstance(args[0], taichi_lang.Expr):
        self.ptr = args[0]
      else:
        self.ptr = taichi_lang.make_constant_expr(args[0])
    else:
      assert False

  def __add__(self, other):
    return Expr(taichi_lang.expr_add(self.ptr, other.ptr))

  def __sub__(self, other):
    return Expr(taichi_lang.expr_sub(self.ptr, other.ptr))

  def __mul__(self, other):
    return Expr(taichi_lang.expr_mul(self.ptr, other.ptr))

  def __div__(self, other):
    return Expr(taichi_lang.expr_div(self.ptr, other.ptr))

  def __le__(self, other):
    return Expr(taichi_lang.expr_cmp_le(self.ptr, other.ptr))

  def __lt__(self, other):
    return Expr(taichi_lang.expr_cmp_lt(self.ptr, other.ptr))

  def __ge__(self, other):
    return Expr(taichi_lang.expr_cmp_ge(self.ptr, other.ptr))

  def __gt__(self, other):
    return Expr(taichi_lang.expr_cmp_gt(self.ptr, other.ptr))

  def __eq__(self, other):
    return Expr(taichi_lang.expr_cmp_eq(self.ptr, other.ptr))

  def __ne__(self, other):
    return Expr(taichi_lang.expr_cmp_ne(self.ptr, other.ptr))

  def serialize(self):
    return self.ptr.serialize()

def main():
  a = Expr(1)
  b = Expr(2)
  c = a + b
  print(c.serialize())

if __name__ == '__main__':
  main()