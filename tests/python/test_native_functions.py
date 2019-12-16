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


@ti.all_archs
def test_minmax():
  x = ti.var(ti.f32)
  y = ti.var(ti.f32)
  minimum = ti.var(ti.f32)
  maximum = ti.var(ti.f32)

  N = 16

  @ti.layout
  def place():
    ti.root.dense(ti.i, N).place(x, y, minimum, maximum)

  @ti.kernel
  def func():
    for i in range(N):
      minimum[i] = min(x[i], y[i])
      maximum[i] = max(x[i], y[i])

  for i in range(N):
    x[i] = i
    y[i] = N - i

  func()

  for i in range(N):
    assert minimum[i] == min(x[i], y[i])
    assert maximum[i] == max(x[i], y[i])
