import taichi as ti


@ti.all_archs
def test_listgen():
  x = ti.var(ti.i32)
  n = 1024
  
  @ti.layout
  def layout():
    ti.root.dense(ti.ij, 4).dense(ti.ij, 4).dense(ti.ij, 4).dense(ti.ij, 4).dense(ti.ij, 4).place(x)
    
  @ti.kernel
  def fill(c: ti.i32):
    for i, j in x:
      x[i, j] = i * 10 + j + c
      
  for c in range(2):
    fill(c)
    
    for i in range(n):
      for j in range(n):
        assert x[i, j] == i * 10 + j + c
      
