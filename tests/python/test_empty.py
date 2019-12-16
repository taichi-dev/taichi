import taichi as ti


@ti.all_archs
def test_abs():

  @ti.kernel
  def func():
    print(1)

  func()
