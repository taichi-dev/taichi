import taichi as ti

# Not really testable..
# Just making sure it does not crash
@ti.all_archs
def print_dt(dt):
  @ti.kernel
  def func():
    print(ti.cast(1234.5, dt))

  func()
  
def test_print():
  for dt in [ti.i32, ti.f32, ti.i64, ti.f64]:
    print_dt(dt)
