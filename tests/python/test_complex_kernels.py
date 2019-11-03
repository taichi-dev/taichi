import taichi as ti

@ti.all_archs
def test_complex_kernels_range():
  a = ti.var(ti.f32)
  b = ti.var(ti.f32)
  ti.cfg.print_ir = True
  
  n = 128
  
  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(a, b)
  
  # Note: the CUDA backend can indeed translate this into multiple kernel launches
  @ti.kernel
  def add():
    for i in range(n):
      a[i] += 1
    for i in range(n):
      b[i] += 2
    for i in range(n):
      b[i] += 3
    for i in range(n):
      a[i] += 1
    for i in range(n):
      a[i] += 9
  
  for i in range(n):
    a[i] = i + 1
    b[i] = i + 2
  add()
  
  for i in range(n):
    assert a[i] == i + 12
    assert b[i] == i + 7

@ti.all_archs
def test_complex_kernels():
  return
  a = ti.var(ti.f32)
  b = ti.var(ti.f32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(a, b)

  # Note: the CUDA backend can indeed translate this into multiple kernel launches
  @ti.kernel
  def add():
    for i in range(n):
      a[i] += 1
    for i in range(n):
      b[i] += 2
    for i in a:
      b[i] += 3
    for i in b:
      a[i] += 1
    for i in a:
      a[i] += 9

  for i in range(n):
    a[i] = i + 1
    b[i] = i + 2
  add()

  for i in range(n):
    assert a[i] == i + 12
    assert b[i] == i + 7

