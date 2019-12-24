import taichi as ti


@ti.all_archs
def test_pointer():
  if ti.get_os_name() == 'win':
    # This test not supported on Windows due to the VirtualAlloc issue #251
    return
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


@ti.all_archs
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
