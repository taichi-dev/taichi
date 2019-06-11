import taichi_lang as ti

def test_size1():
  x = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x)

  x[0] = 1
  assert x[0] == 1

test_size1()

def test_diff_sincos():
  x = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x, x.grad)

  x[0] = 1
  assert x[0] == 1
