import taichi as ti

@ti.all_archs
def test_dynamic():
  x = ti.var(ti.f32)
  n = 128

  @ti.layout
  def place():
    ti.root.dynamic(ti.i, n).place(x)

  @ti.kernel
  def func():
    pass
  
  for i in range(n):
    x[i] = i
    
  for i in range(n):
    assert x[i] == i
