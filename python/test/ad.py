import taichi_lang as ti
import math

def test_size1():
  ti.reset()
  x = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x)

  x[0] = 1
  assert x[0] == 1

# test_size1()

def test_diff_sincos():
  ti.reset()

  x = ti.var(ti.f32)
  y = ti.var(ti.f32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x, x.grad, y, y.grad)

  @ti.kernel
  def tri():
    for i in x:
      y[i] = ti.sin(x[i])

  v = 1
  x[0] = v
  tri()
  tri.grad()
  print(y[0], math.sin(v))
  print(x.grad[0], math.cos(v))

test_diff_sincos()
