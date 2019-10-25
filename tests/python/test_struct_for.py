import taichi as ti


@ti.program_test
def test_linear():
  x = ti.var(ti.i32)
  y = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(x)
    ti.root.dense(ti.i, n).place(y)

  @ti.kernel
  def fill():
    for i in x:
      x[i] = i
      y[i] = i * 2
      
  fill()

  for i in range(n):
    assert x[i] == i
    assert y[i] == i * 2
