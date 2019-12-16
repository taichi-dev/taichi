import taichi as ti


@ti.all_archs
def test_cond_grad():
  x = ti.var(ti.f32)
  y = ti.var(ti.f32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 2).place(x, x.grad, y, y.grad)

  @ti.kernel
  def func():
    for i in range(2):
      t = 0.0
      if x[i] > 0:
        t = 1 / (x[i] + 1e-10)
      y[i] = t

  x[0] = 0
  x[1] = 1
  y.grad[0] = 1
  y.grad[1] = 1

  func()
  func.grad()

  assert x.grad[0] == 0
  assert x.grad[1] == -1
