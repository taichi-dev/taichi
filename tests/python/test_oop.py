import taichi as ti

@ti.host_arch
def test_oop():

  class Array2D:
    def __init__(self, n, m, increment):
      self.n = n
      self.m = m
      self.val = ti.var(ti.i32)
      self.increment = increment

    def layout(self, root):
      root.dense(ti.ij, (self.n, self.m)).place(self.val)

    @ti.kernel
    def inc(self: ti.template()):
      for i, j in self.val:
        self.val[i, j] += self.increment

  arr = Array2D(128, 128, 3)

  @ti.layout
  def place():
    arr.layout(ti.root)

  ti.get_runtime().materialize()

  arr.inc(arr)
  print(arr.val[3, 4])
