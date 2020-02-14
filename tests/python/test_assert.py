import taichi as ti

@ti.all_archs
def test_assert():
  return
  
  @ti.kernel
  def func():
    x = 20
    assert 10 <= x < 20
    
  func()
