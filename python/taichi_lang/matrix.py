import taichi_lang.impl as impl

class Matrix:
  is_taichi_class = True

  def __init__(self, n, m, dt=None, empty=False):
    self.entries = []
    self.n = n
    self.m = m
    if empty:
      self.entries = [None] * n * m
    else:
      if dt is None:
        for i in range(n * m):
          self.entries.append(impl.expr_init(0))
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

  def __div__(self, other):
    assert self.n == other.n and self.m == other.m
    ret = Matrix(self.n, self.m)
    for i in range(self.n):
      for j in range(self.m):
        ret(i, j).assign(self(i, j) / other(i, j))
    return ret

  def __mul__(self, other):
    assert self.n == other.n and self.m == other.m
    ret = Matrix(self.n, self.m)
    for i in range(self.n):
      for j in range(self.m):
        ret(i, j).assign(self(i, j) * other(i, j))
    return ret

  def __add__(self, other):
    assert self.n == other.n and self.m == other.m
    ret = Matrix(self.n, self.m)
    for i in range(self.n):
      for j in range(self.m):
        ret(i, j).assign(self(i, j) + other(i, j))
    return ret

  __radd__ = __add__

  def __sub__(self, other):
    assert self.n == other.n and self.m == other.m
    ret = Matrix(self.n, self.m)
    for i in range(self.n):
      for j in range(self.m):
        ret(i, j).assign(self(i, j) - other(i, j))
    return ret

  def __call__(self, *args, **kwargs):
    assert kwargs == {}
    assert 1 <= len(args) <= 2
    if len(args) == 1:
      args.append(0)
    assert 0 <= args[0] < self.n
    assert 0 <= args[1] < self.m
    return self.entries[args[0] * self.n + args[1]]

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

  def __setitem__(self, item):
    assert False
