import taichi as ti


'''
ti.cfg.use_llvm = True
def test_linear():
  ti.reset()

  x = ti.var(ti.i32)
  y = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(x)
    ti.root.dense(ti.i, n).place(y)

  for i in range(n):
    x[i] = i
    y[i] = i + 123

  for i in range(n):
    assert x[i] == i
    assert y[i] == i + 123

def test_linear_nested():
  ti.reset()

  x = ti.var(ti.i32)
  y = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(x)
    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(y)

  for i in range(n):
    x[i] = i
    y[i] = i + 123

  for i in range(n):
    assert x[i] == i
    assert y[i] == i + 123

def test_2d_nested():
  ti.reset()

  x = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.ij, n // 16).dense(ti.ij, (32, 16)).place(x)

  for i in range(n * 2):
    for j in range(n):
      x[i, j] = i + j * 10

  for i in range(n * 2):
    for j in range(n):
      assert x[i, j] == i + j * 10

'''
