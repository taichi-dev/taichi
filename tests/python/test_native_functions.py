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
      ti.static_print(x[i])

  func()

  for i in range(N):
    assert x[i] == i


@ti.all_archs
def test_int():
  x = ti.var(ti.f32)

  N = 16

  @ti.layout
  def place():
    ti.root.dense(ti.i, N).place(x)

  @ti.kernel
  def func():
    for i in range(N):
      x[i] = int(x[i])
      x[i] = float(int(x[i]) // 2)


  for i in range(N):
    x[i] = i + 0.4

  func()

  for i in range(N):
    assert x[i] == i // 2
