import taichi as ti
from pytest import approx
from autograd import grad


def grad_test(tifunc, npfunc=None):
  if npfunc is None:
    npfunc = tifunc
  ti.reset()

  x = ti.var(ti.f32)
  y = ti.var(ti.f32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x, x.grad, y, y.grad)

  @ti.kernel
  def func():
    for i in x:
      y[i] = tifunc(x[i])

  v = 0.2

  y.grad[0] = 1
  x[0] = v
  func()
  func.grad()

  assert y[0] == approx(npfunc(v))
  assert x.grad[0] == approx(grad(npfunc)(v))
