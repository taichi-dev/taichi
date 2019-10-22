import taichi as ti

@ti.program_test
def test_loops():
  x = ti.var(ti.f32)
  y = ti.var(ti.f32)

  N = 512
  @ti.layout
  def place():
    ti.root.dense(ti.i, N).place(x)
    ti.root.dense(ti.i, N).place(y)
    ti.root.lazy_grad()

  for i in range(N // 2, N):
    y[i] = i - 300

  @ti.kernel
  def func():
    for i in range(ti.static(N // 2 + 3), N):
      x[i] = ti.abs(y[i])

  func()

  for i in range(N // 2 + 3):
    assert x[i] == 0

  for i in range(N // 2 + 3, N):
    assert x[i] == abs(y[i])

@ti.program_test
def test_numpy_loops():
  x = ti.var(ti.f32)
  y = ti.var(ti.f32)

  N = 512
  @ti.layout
  def place():
    ti.root.dense(ti.i, N).place(x)
    ti.root.dense(ti.i, N).place(y)
    ti.root.lazy_grad()

  for i in range(N // 2, N):
    y[i] = i - 300

  import numpy as np
  begin = (np.ones(1) * (N // 2 + 3)).astype(np.int32)
  end = (np.ones(1) * N).astype(np.int32)

  @ti.kernel
  def func():
    for i in range(begin, end):
      x[i] = ti.abs(y[i])

  func()

  for i in range(N // 2 + 3):
    assert x[i] == 0

  for i in range(N // 2 + 3, N):
    assert x[i] == abs(y[i])

