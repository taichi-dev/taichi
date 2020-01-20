import taichi as ti

@ti.all_archs
def test_basic():
  @ti.kernel
  def test():
    ti.call_internal("do_nothing")
    
  test()

