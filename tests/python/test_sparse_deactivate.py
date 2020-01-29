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

@ti.all_archs
def test_dynamic():
  x = ti.var(ti.i32)
  s = ti.var(ti.i32)

  n = 16

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).dynamic(ti.j, 4096).place(x)
    ti.root.dense(ti.i, n).place(s)

  @ti.kernel
  def func(mul: ti.i32):
    for i in range(n):
      for j in range(i * i * mul):
        ti.append(x.parent(), i, j)
      s[i] = ti.length(x.parent(), i)


  func(1)

  for i in range(n):
    assert s[i] == i * i

  @ti.kernel
  def clear():
    for i in range(n):
      ti.deactivate(x.parent(), i)

  func(2)

  for i in range(n):
    assert s[i] == i * i * 3

  clear()
  func(4)

  for i in range(n):
    assert s[i] == i * i * 4

