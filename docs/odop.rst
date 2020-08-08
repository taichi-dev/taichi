Objective data-oriented programming
===================================

Taichi is a `data-oriented <https://en.wikipedia.org/wiki/Data-oriented_design>`_ programming (DOP) language. However, simple DOP makes modularization hard.

To allow modularized code, Taichi borrow some concepts from object-oriented programming (OOP).

For convenience, let's call the hybrid scheme **objective data-oriented programming** (ODOP).

TODO: More documentation here.

A brief example:

.. code-block:: python

  import taichi as ti

  ti.init()

  @ti.data_oriented
  class Array2D:
    def __init__(self, n, m, increment):
      self.n = n
      self.m = m
      self.val = ti.field(ti.f32)
      self.total = ti.field(ti.f32)
      self.increment = increment
      ti.root.dense(ti.ij, (self.n, self.m)).place(self.val)
      ti.root.place(self.total)

    @staticmethod
    @ti.func
    def clamp(x):  # Clamp to [0, 1)
        return max(0, min(1 - 1e-6, x))

    @ti.kernel
    def inc(self):
      for i, j in self.val:
        ti.atomic_add(self.val[i, j], self.increment)

    @ti.kernel
    def inc2(self, increment: ti.i32):
      for i, j in self.val:
        ti.atomic_add(self.val[i, j], increment)

    @ti.kernel
    def reduce(self):
      for i, j in self.val:
        ti.atomic_add(self.total, self.val[i, j] * 4)

  arr = Array2D(128, 128, 3)

  double_total = ti.field(ti.f32, shape=())

  ti.root.lazy_grad()

  arr.inc()
  arr.inc.grad()
  assert arr.val[3, 4] == 3
  arr.inc2(4)
  assert arr.val[3, 4] == 7

  with ti.Tape(loss=arr.total):
    arr.reduce()

  for i in range(arr.n):
    for j in range(arr.m):
      assert arr.val.grad[i, j] == 4

  @ti.kernel
  def double():
    double_total[None] = 2 * arr.total

  with ti.Tape(loss=double_total):
    arr.reduce()
    double()

  for i in range(arr.n):
    for j in range(arr.m):
      assert arr.val.grad[i, j] == 8
