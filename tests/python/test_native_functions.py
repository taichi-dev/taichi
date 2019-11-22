import taichi as ti

@ti.all_archs
def test_abs():
  x = ti.var(ti.f32)

  N = 16
  @ti.layout
  def place():
    ti.root.dense(ti.i, N).place(x)

  @ti.kernel
  def func():
    for i in range(N):
      x[i] = abs(-i)
      print(x[i])

  func()

  for i in range(N):
    assert x[i] == i
