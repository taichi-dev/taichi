import taichi as ti

@ti.all_archs
def test_pointer():
  x = ti.var(ti.f32)
  s = ti.var(ti.i32)

  n = 16

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).pointer().dense(ti.i, n).place(x)
    ti.root.place(s)

  s[None] = 0

  @ti.kernel
  def func():
    for i in x:
      s[None] += 1


  x[0] = 1
  x[19] = 1
  func()
  assert s[None] == 32

  @ti.kernel
  def deactivate():
    ti.deactivate(x.parent().parent(), 4)

  deactivate()
  s[None] = 0
  func()
  assert s[None] == 16
