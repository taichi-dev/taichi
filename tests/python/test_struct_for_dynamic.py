import taichi as ti

@ti.all_archs
def test_dynamic():
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


@ti.all_archs
def test_dense_dynamic():
  n = 128
  
  x = ti.var(ti.i32)
  y = ti.var(ti.i32, shape=(n))
  
  @ti.layout
  def place():
    ti.root.dense(ti.i, n).dynamic(ti.j, n, 128).place(x)
  
  @ti.kernel
  def append():
    for i in range(n // 2):
      for j in range(i * 2):
        ti.append(x, i, j * 2)
    
  append()
  
  @ti.kernel
  def get_len():
    for i in range(n // 2):
      y[i] = ti.length(x, i)
      
  get_len()
  for i in range(n // 2):
    assert y[i] == i * 2
    y[i] = 0
  
  @ti.kernel
  def count():
    ti.serialize()
    for i, j in x:
      print(i)
      print(j)
      print(x[i, j])
      assert x[i, j] == j * 2
      y[i] += x[i, j]
  
  count()
  
  for i in range(n):
    print(i)
    assert y[i] == i * (i - 1)
