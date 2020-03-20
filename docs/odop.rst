Objective data-oriented programming
====================================================

Taichi is a `data-oriented <https://en.wikipedia.org/wiki/Data-oriented_design>`_ programming (DOP) language. However, simple DOP makes modularization hard.

To allow modularized code, Taichi borrow some concepts from object-oriented programming (OOP).

For convenience, let's call the hybrid scheme **objective data-oriented programming** (ODOP).

TODO: More documentation here.

A brief example:

.. code-block:: python

  @ti.data_oriented
  class Array2D:
    def __init__(self, n, m, increment):
      self.n = n
      self.m = m
      self.val = ti.var(ti.f32)
      self.total = ti.var(ti.f32)
      self.increment = increment

    @staticmethod
    @ti.func
    def clamp(x):  # Clamp to [0, 1)
        return max(0, min(1 - 1e-6, x))

    def place(self, root):
      root.dense(ti.ij, (self.n, self.m)).place(self.val)
      root.place(self.total)

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

  double_total = ti.var(ti.f32)

  @ti.layout
  def place():
    ti.root.place(
        arr)  # Place an object. Make sure you defined place for that obj
    ti.root.place(double_total)
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
