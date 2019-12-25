import taichi as ti

# @ti.all_archs
def test_assert():
  ti.cfg.arch = ti.cuda
  
  @ti.kernel
  def func():
    assert False
    
  func()

