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
      else:
        mat = n
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
            self.entries.append(impl.expr_init(None))
        else:
          assert not impl.inside_kernel()
          for i in range(n * m):
            self.entries.append(impl.var(dt))

  def assign(self, other):
    if not isinstance(other, Matrix):
      other = Matrix(other)
    assert other.n == self.n and other.m == self.m
    for i in range(self.n * self.m):
      self.entries[i].assign(other.entries[i])

  def __matmul__(self, other):
    assert self.m == other.n
    ret = Matrix(self.n, other.m)
    for i in range(self.n):
      for j in range(other.m):
        ret(i, j).assign(self(i, 0) * other(0, j))
        for k in range(1, other.n):
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

  def linearize_entry_id(self, *args):
    assert 1 <= len(args) <= 2
    if len(args) == 1:
      args = args + (0,)
    assert 0 <= args[0] < self.n
    assert 0 <= args[1] < self.m
    return args[0] * self.m + args[1]

  def __call__(self, *args, **kwargs):
    assert kwargs == {}
    return self.entries[self.linearize_entry_id(*args)]

  def get_entry(self, *args, **kwargs):
    assert kwargs == {}
    return self.entries[self.linearize_entry_id(*args)]

  def set_entry(self, i, j, e):
    self.entries[self.linearize_entry_id(i, j)] = e

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

  def trace(self):
    assert self.n == self.m
    sum = self(0, 0)
    for i in range(1, self.n):
      sum = sum + self(i, i)
    return sum

  @staticmethod
  def outer_product(a, b):
    assert a.m == 1
    assert b.m == 1
    c = Matrix(a.n, b.n)
    for i in range(a.n):
      for j in range(b.n):
        c(i, j).assign(a(i) * b(j))
    return c

  @staticmethod
  def transposed(a):
    assert a.n == a.m
    ret = Matrix(a.n, a.m, empty=True)
    for i in range(a.n):
      for j in range(a.m):
        ret.set_entry(j, i, a(i, j))
    return ret

  @staticmethod
  def polar_decomposition(a):
    assert a.n == 2 and a.m == 2
    x, y = a(0, 0) + a(11), a(1, 0) - a(0, 1)
    scale = impl.expr_init(1.0 / impl.sqrt(x * x + y * y))
    c = x * scale
    s = y * scale
    r = Matrix([[c, -s], [s, c]])
    return r, Matrix.transposed(r) @ a

  def loop_range(self):
    return self.entries[0]

  def augassign(self, other, op):
    if not isinstance(other, Matrix):
      other = Matrix(other)
    assert self.n == other.n and self.m == other.m
    for i in range(len(self.entries)):
      self.entries[i].augassign(other.entries[i], op)

  def atomic_add(self, other):
    assert self.n == other.n and self.m == other.m
    for i in range(len(self.entries)):
      self.entries[i].atomic_add(other.entries[i])
