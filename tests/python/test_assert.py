import taichi as ti

@ti.all_archs
def test_assert():
  ti.cfg.print_ir = True
  
  @ti.kernel
  def func():
    assert False
    
  func()

