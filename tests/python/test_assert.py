import taichi as ti

@ti.all_archs
def test_assert():
  return
  ti.cfg.arch = ti.cuda
  
  @ti.kernel
  def func():
    assert 1 + 1 == 3
    assert False
    
  func()
