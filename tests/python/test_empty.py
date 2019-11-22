import taichi as ti

@ti.all_archs
def test_abs():
  @ti.kernel
  def func():
    ti.print(1)

  func()
