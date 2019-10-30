import taichi as ti

@ti.program_test
def test_while():
  x = ti.var(ti.f32)
  s = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).bitmasked().dense(ti.i, n).place(x)
    ti.root.place(s)

  @ti.kernel
  def func():
    for i in x:
      ti.atomic_add(s[None], 1)

  x[0] = 1

  func()
  assert s[None] == 128
