import taichi as ti

@ti.all_archs
def test_empty():

  @ti.kernel
  def func():
    pass

  func()
