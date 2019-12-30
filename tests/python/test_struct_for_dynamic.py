import taichi as ti

@ti.all_archs
def test_dynamic():
  x = ti.var(ti.i32)
  y = ti.var(ti.i32, shape=())
  # ti.cfg.print_ir = True
  # ti.cfg.print_kernel_llvm_ir = True

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
    ti.root.dense(ti.i, n).dynamic(ti.j, n * 2).place(x)
  
  @ti.kernel
  def append():
    for i in range(n):
      for j in range(i):
        ti.append(x, i, j * 2)
    
  append()
  
  @ti.kernel
  def count():
    for i, j in x:
      y[i] += x[i, j]
  
  count()
  
  for i in range(n):
    print(i)
    assert y[i] == i * (i - 1)
