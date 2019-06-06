import taichi_lang as ti

class Matrix:
  def __init__(self, n, m, dt=None):
    self.entries = []
    self.n = n
    self.m = m
    if dt is None:
      for i in range(n * m):
        self.entries.append(ti.expr_init(0.0))
    else:
      for i in range(n * m):
        self.entries.append(ti.var(dt))


  def assign(self, other):
    assert other.n == self.n and other.m == self.m
    for i in range(self.n * self.m):
      self.entries[i].assign(other.entries[i])

  def __mul__(self, other):
    assert self.m == other.n
    ret = Matrix(self.n, other.m)
    for i in range(self.n):
      for j in range(self.m):
        for k in range(other.m):
          ret(i, j).assign(ret(i, j) + self(i, k) * other(k, j))
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

x = Matrix(2, 2, ti.i32)

@ti.layout
def xy():
  ti.root.dense(ti.i, 16).place(x)

@ti.kernel
def inc():
  for i in x(0, 0):
    x(1, 1)[i] = x(0, 0)[i] + 1

for i in range(10):
  x(0, 0)[i] = i

inc()

for i in range(10):
  print(x(1, 1)[i])



