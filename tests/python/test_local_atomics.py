import taichi as ti


@ti.all_archs
def test_explicit_local_atomics():
  A = ti.var(ti.f32, shape=())

  @ti.kernel
  def func():
    a = 0
    for i in range(10):
      ti.atomic_add(a, i)
    A[None] = a

  func()
  assert A[None] == 45


@ti.all_archs
def test_implicit_local_atomics():
  A = ti.var(ti.f32, shape=())

  @ti.kernel
  def func():
    a = 0
    for i in range(10):
      a += i
    A[None] = a

  func()
  assert A[None] == 45
