import taichi as ti
from pytest import approx


@ti.all_archs
def test_ad_reduce():
  x = ti.var(ti.f32)
  loss = ti.var(ti.f32)

  N = 16

  @ti.layout
  def place():
    ti.root.place(loss, loss.grad).dense(ti.i, N).place(x, x.grad)

  @ti.kernel
  def func():
    for i in x:
      loss.atomic_add(ti.sqr(x[i]))

  total_loss = 0
  for i in range(N):
    x[i] = i
    total_loss += i * i

  loss.grad[None] = 1
  func()
  func.grad()

  assert total_loss == approx(loss[None])
  for i in range(N):
    assert x.grad[i] == approx(i * 2)
