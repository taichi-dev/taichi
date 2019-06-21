import taichi_lang as ti
from pytest import approx

def test_abs():
  x = ti.var(ti.f32)
  y = ti.var(ti.f32)

  N = 16
  @ti.layout
  def place():
    ti.root.dense(ti.i, N).place(x)
    ti.root.dense(ti.i, N).place(y)
    ti.root.lazy_grad()

  @ti.kernel
  def func():
    for i in range(N):
      x[i] = ti.abs(y[i])

  for i in range(N):
    y[i] = i - 10
  func()

  for i in range(N):
    assert x[i] == abs(y[i])

test_abs()
