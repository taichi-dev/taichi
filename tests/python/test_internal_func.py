import taichi as ti

@ti.all_archs
def test_basic():
  ti.cfg.print_ir = True
  
  @ti.kernel
  def test():
    for i in range(10):
      ti.call_internal("do_nothing")
    
  test()

