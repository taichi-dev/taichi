import taichi as ti

@ti.all_archs
def test_dynamic():
  return
  x = ti.var(ti.i32)
  y = ti.var(ti.i32, shape=())

  n = 128

  @ti.layout
  def place():
    ti.root.dynamic(ti.i, n).place(x)

  @ti.kernel
  def count():
    for i in x:
      y[None] += 1
      
  x[n // 3] = 1

  count()

  assert y[None] == n // 3 + 1


# @ti.all_archs
def test_dense_dynamic():
  n = 128
  
  x = ti.var(ti.i32)
  
  @ti.layout
  def place():
    ti.root.dense(ti.i, n).dynamic(ti.j, n, 128).place(x)
  
  @ti.kernel
  def append():
    ti.serialize()
    for i in range(33, 36):
      for j in range(66):
        ti.append(x, i, j * 2)
        # x[i, j] = j * 2
    
  append()
  
  print(x[33, 65])
  assert x[33, 65] == 130

test_dense_dynamic()
