import taichi as ti


@ti.all_archs
def test_simple():
  x = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(x)

  @ti.kernel
  def func():
    x[7] = 120

  func()

  for i in range(n):
    if i == 7:
      assert x[i] == 120
    else:
      assert x[i] == 0


@ti.all_archs
def test_range_loops():
  x = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(x)

  @ti.kernel
  def func():
    for i in range(n):
      x[i] = i + 123

  func()

  for i in range(n):
    assert x[i] == i + 123


@ti.all_archs
def test_io():
  ti.cfg.arch = ti.cuda
  x = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(x)

  x[3] = 123
  x[4] = 456
  assert x[3] == 123
  assert x[4] == 456


@ti.all_archs
def test_if():
  x = ti.var(ti.f32)

  @ti.layout
  def xy():
    ti.root.dense(ti.i, 16).place(x)

  @ti.kernel
  def if_test():
    for i in x:
      if i < 100:
        x[i] = 100
      else:
        x[i] = i

  if_test()

  for i in range(16):
    assert x[i] == 100

  @ti.kernel
  def if_test2():
    for i in x:
      if i < 100:
        x[i] = i
      else:
        x[i] = 100

  if_test2()

  for i in range(16):
    assert x[i] == i
