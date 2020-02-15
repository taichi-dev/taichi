import taichi as ti


@ti.all_archs
def test_pow_scan():
  z = ti.var(ti.f32, shape=())
  w = ti.var(ti.f32, shape=())
  
  @ti.kernel
  def func(x: ti.f32, y: ti.f32):
    z[None] = x ** y
  
  for x in [2, 1.4, 3, 5.11, 6.12, 10.0]:
    for y in [2, 0.5, 4, 8, 1.12, -1, -9, 2.01, -3.3]:
      func(x, y)
      assert abs(z[None] / x ** y - 1) < 0.0001
