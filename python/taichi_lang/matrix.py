import taichi_lang.impl as impl
import taichi_lang.expr as expr
import copy
import numbers


def broadcast_if_scalar(func):
  def broadcasted(self, other):
    if isinstance(other, expr.Expr) or isinstance(other, numbers.Number):
      other = self.broadcast(expr.Expr(other))
    return func(self, other)

  return broadcasted


class Matrix:
  is_taichi_class = True

  def __init__(self, n, m=1, dt=None, empty=False):
    if isinstance(n, list):
      if not isinstance(n[0], list):
        mat = [list([expr.Expr(x)]) for x in n]
      self.n, self.m = len(mat), len(mat[0])
      self.entries = [x for row in mat for x in row]
    else:
      self.entries = []
      self.n = n
      self.m = m
      self.dt = dt
      if empty:
        self.entries = [None] * n * m
      else:
        if dt is None:
          for i in range(n * m):
            self.entries.append(impl.expr_init(0.0))
        else:
          assert not impl.inside_kernel()
          for i in range(n * m):
            self.entries.append(impl.var(dt))

  def assign(self, other):
    assert other.n == self.n and other.m == self.m
    for i in range(self.n * self.m):
      self.entries[i].assign(other.entries[i])

  def __matmul__(self, other):
    assert self.m == other.n
    ret = Matrix(self.n, other.m)
    for i in range(self.n):
      for j in range(self.m):
        for k in range(other.m):
          ret(i, j).assign(ret(i, j) + self(i, k) * other(k, j))
    return ret

  @broadcast_if_scalar
  def __div__(self, other):
    assert self.n == other.n and self.m == other.m
    ret = Matrix(self.n, self.m)
    for i in range(self.n):
      for j in range(self.m):
        ret(i, j).assign(self(i, j) / other(i, j))
    return ret

  def broadcast(self, scalar):
    ret = Matrix(self.n, self.m, empty=True)
    for i in range(self.n * self.m):
      ret.entries[i] = scalar
    return ret

  @broadcast_if_scalar
  def __mul__(self, other):
    assert self.n == other.n and self.m == other.m
    ret = Matrix(self.n, self.m)
    for i in range(self.n):
      for j in range(self.m):
        ret(i, j).assign(self(i, j) * other(i, j))
    return ret

  __rmul__ = __mul__

  @broadcast_if_scalar
  def __add__(self, other):
    assert self.n == other.n and self.m == other.m
    ret = Matrix(self.n, self.m)
    for i in range(self.n):
      for j in range(self.m):
        ret(i, j).assign(self(i, j) + other(i, j))
    return ret

  __radd__ = __add__

  @broadcast_if_scalar
  def __sub__(self, other):
    assert self.n == other.n and self.m == other.m
    ret = Matrix(self.n, self.m)
    for i in range(self.n):
      for j in range(self.m):
        ret(i, j).assign(self(i, j) - other(i, j))
    return ret

  @broadcast_if_scalar
  def __rsub__(self, other):
    assert self.n == other.n and self.m == other.m
    ret = Matrix(self.n, self.m)
    for i in range(self.n):
      for j in range(self.m):
        ret(i, j).assign(other(i, j) - self(i, j))
    return ret

  def __call__(self, *args, **kwargs):
    assert kwargs == {}
    assert 1 <= len(args) <= 2
    if len(args) == 1:
      args = args + (0,)
    assert 0 <= args[0] < self.n
    assert 0 <= args[1] < self.m
    return self.entries[args[0] * self.m + args[1]]

  def place(self, snode):
    for e in self.entries:
      snode.place(e)

  def subscript(self, *indices):
    ret = Matrix(self.n, self.m, empty=True)
    for i, e in enumerate(self.entries):
      ret.entries[i] = impl.subscript(e, *indices)
    return ret

  def __getitem__(self, item):
    assert False

  def __setitem__(self, index, item):
    if not isinstance(item[0], list):
      item = [[i] for i in item]
    for i in range(self.n):
      for j in range(self.m):
        self(i, j)[index] = item[i][j]

  def copy(self):
    ret = Matrix(self.n, self.m)
    ret.entries = copy.copy(self.entries)
    return ret

  def variable(self):
    ret = self.copy()
    ret.entries = [impl.expr_init(e) for e in ret.entries]
    return ret

  def cast(self, type):
    ret = self.copy()
    for i in range(len(self.entries)):
      ret.entries[i] = impl.cast(ret.entries[i], type)
    return ret


