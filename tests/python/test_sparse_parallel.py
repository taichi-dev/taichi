import taichi as ti

@ti.all_archs
def test_pointer():
  x = ti.var(ti.f32)
  s = ti.var(ti.i32)
  
  n = 128
  
  @ti.layout
  def place():
    ti.root.dense(ti.i, n).pointer().dense(ti.i, n).place(x)
    ti.root.place(s)
    
  @ti.kernel
  def activate():
    for i in range(n):
      x[i * n] = 0
  
  @ti.kernel
  def func():
    for i in x:
      ti.atomic_add(s[None], 1)
  
  
  activate()
  func()
  assert s[None] == n * n


@ti.all_archs
def test_pointer2():
  ti.cfg.verbose_kernel_launches = True
  x = ti.var(ti.f32)
  s = ti.var(ti.i32)
  
  n = 128
  
  @ti.layout
  def place():
    ti.root.dense(ti.i, n).pointer().dense(ti.i, n).place(x)
    ti.root.place(s)
  
  @ti.kernel
  def activate():
    for i in range(n * n):
      x[i] = i
  
  @ti.kernel
  def func():
    for i in x:
      ti.atomic_add(s[None], i)
  
  
  activate()
  func()
  N = n * n
  assert s[None] == N * (N - 1) / 2
