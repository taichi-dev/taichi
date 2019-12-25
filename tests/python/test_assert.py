import taichi as ti

@ti.all_archs
def test_assert():
  return
  ti.get_runtime().print_preprocessed = True
  ti.cfg.print_ir = True
  # ti.cfg.arch = ti.cuda
  
  @ti.kernel
  def func():
    x = 20
    assert 10 <= x < 20
    
  func()
