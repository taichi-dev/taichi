import taichi as ti


@ti.all_archs
def _test_floor_div(arg1, a, arg2, b, arg3, c):
  z = ti.var(arg3, shape=())

  @ti.kernel
  def func(x: arg1, y: arg2):
    z[None] = x // y

  func(a, b)
  assert z[None] == c

  
def test_floor_div():
  _test_floor_div(ti.i32, 10, ti.i32, 3, ti.i32, 3)
  _test_floor_div(ti.f32, 10, ti.f32, 3, ti.i32, 3)
  _test_floor_div(ti.i32, 10, ti.f32, 3, ti.i32, 3)
  _test_floor_div(ti.f32, 10, ti.i32, 3, ti.i32, 3)
