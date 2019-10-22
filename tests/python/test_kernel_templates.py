import taichi as ti

def test_kernel_template_mapper():
  ti.reset()
  x = ti.var(ti.i32)
  y = ti.var(ti.f32)
  
  n = 16
  
  @ti.layout
  def layout():
    ti.root.dense(ti.i, n).place(x, y)
  
  @ti.kernel
  def inc(a: ti.template(), b: ti.template()):
    for i in a:
      a[i] += b
  
  inc(x, 1)
  inc(y, 2)
  
  for i in range(n):
    assert x[i] == 1
    assert y[i] == 2
