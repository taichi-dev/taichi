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

    @ti.classkernel
    def inc(self: ti.template()):
      for i, j in self.val:
        self.val[i, j] += self.increment

  arr = Array2D(128, 128, 3)

  @ti.layout
  def place():
    ti.root.place(arr)

  print(arr.inc)
  arr.inc()
  assert arr.val[3, 4] == 3
  arr.inc()
  assert arr.val[3, 4] == 6


def wrap(f):
  def wrapped(self):
    print(self)
    f(self)

  return wrapped

def wrap2(f):
  def decorated(self1, *args, **kwargs):
    class F:
      def __call__(self, *args, **kwargs):
        print(self)
        print(args)
        print(kwargs)
        f(self1, *args, **kwargs)

    FF = F()
    return FF(*args, **kwargs)
  return decorated

class A:
  def __init__(self):
    self.val = 123

  @wrap2
  def test(self: ti.template()):
    print(self.val)

a = A()
a.test()
