import taichi as ti


@ti.all_archs
def test_running_loss():
  return
  steps = 16

  total_loss = ti.var(ti.f32)
  running_loss = ti.var(ti.f32)
  additional_loss = ti.var(ti.f32)

  @ti.layout
  def place():
    ti.root.place(total_loss)
    ti.root.dense(ti.i, steps).place(running_loss)
    ti.root.place(additional_loss)
    ti.root.lazy_grad()

  @ti.kernel
  def compute_loss():
    total_loss[None] = 0.0
    for i in range(steps):
      total_loss[None].atomic_add(running_loss[i] * 2)
    total_loss[None].atomic_add(additional_loss[None] * 3)

  compute_loss()

  assert total_loss.grad[None] == 1
  for i in range(steps):
    assert running_loss[i] == 2
  assert additional_loss.grad[None] == 3
