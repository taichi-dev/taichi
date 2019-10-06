import taichi as ti
from pytest import approx

def test_clear():
  return
  ti.reset()
  x = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x)

  x[0] = 1
  assert x[0] == 1
  x.clear()
  assert x[0] == 0

def test_nested_subscript():
  ti.reset()

  x = ti.var(ti.i32)
  y = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x)
    ti.root.dense(ti.i, 1).place(y)

  x[0] = 0

  @ti.kernel
  def inc():
    for i in range(1):
      x[x[i]] += 1

  inc()

  assert x[0] == 1
