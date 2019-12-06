import taichi as ti

@ti.all_archs
def test_explicit_local_atomics():
  ti.cfg.print_ir = True
  ti.get_runtime().print_preprocessed = True
  A = ti.var(ti.i32, shape=())

  @ti.kernel
  def func():
    a = 0
    for i in range(10):
      ti.atomic_add(a, i)
    A[None] = a


  func()
  assert A[None] == 45
