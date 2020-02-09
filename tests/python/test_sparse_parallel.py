import taichi as ti


@ti.archs_support_sparse
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
      s[None] += 1

  activate()
  func()
  assert s[None] == n * n


@ti.archs_support_sparse
def test_pointer2():
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
      s[None] += i

  activate()
  func()
  N = n * n
  assert s[None] == N * (N - 1) / 2

@ti.archs_support_sparse
def test_nested_struct_fill_and_clear():
  a = ti.var(dt=ti.f32)
  N = 512

  @ti.layout
  def place():
    ti.root.dense(ti.ij, [N, N]).pointer().dense(ti.ij, [8, 8]).place(a)

  @ti.kernel
  def fill():
    for i, j in ti.ndrange(N * 8, N * 8):
      a[i, j] = 2.0

  @ti.kernel
  def clear():
    for i, j in a.parent():
      ti.deactivate(a.parent().parent(), [i, j])


  def task():
    fill()
    clear()

  for i in range(10):
    task()
    ti.sync()

