import taichi as ti

@ti.all_archs
def test_cse():
  A = ti.var(ti.f32, shape=())

  @ti.kernel
  def func():
    a = 0
    a += 10
    a = a + 123
    A[None] = a

  func()
  assert A[None] == 133

