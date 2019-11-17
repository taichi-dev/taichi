import taichi as ti
import inspect

@ti.host_arch
def test_oop():

  class Array2D:
    def __init__(self, n, m, increment):
      self.n = n
      self.m = m
      self.val = ti.var(ti.i32)
      self.increment = increment

    def place(self, root):
      root.dense(ti.ij, (self.n, self.m)).place(self.val)

    @ti.kernel
    def inc(self: ti.template()):
      for i, j in self.val:
        self.val[i, j] += self.increment

  arr = Array2D(128, 128, 3)

  @ti.layout
  def place():
    ti.root.place(arr)

  arr.inc(arr)
  assert arr.val[3, 4] == 3
  arr.inc(arr)
  assert arr.val[3, 4] == 6
