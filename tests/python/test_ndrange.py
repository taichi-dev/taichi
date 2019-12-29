import taichi as ti

@ti.all_archs
def test_1d():
  x = ti.var(ti.f32, shape=(16))

  @ti.kernel
  def func():
    for i in ti.ndrange((4, 10)):
      x[i] = i

  func()
  for i in range(16):
    if 4 <= i < 10:
      assert x[i] == i
    else:
      assert x[i] == 0

