import taichi as ti

@ti.all_archs
def test_cast():
  z = ti.var(ti.i32, shape=())
  
  @ti.kernel
  def func():
    z[None] = ti.cast(1e13, ti.f64) / ti.cast(1e10, ti.f64) + 1e-3
  
  func()
  assert z[None] == 1000
