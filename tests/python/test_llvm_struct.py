import taichi as ti

def test_linear():
  ti.reset()
  ti.core.test_throw()
  ti.cfg.use_llvm = True

  x = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.place(x)

  @ti.kernel
  def func():
    x[0] = 0

  ti.core.test_throw()
  func()
  ti.core.test_throw()
  
for i in range(2):
  test_linear()
    

