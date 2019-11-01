import taichi as ti

@ti.program_test
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

@ti.program_test
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


@ti.program_test
def test_io():
  ti.cfg.print_ir = True
  x = ti.var(ti.i32)
  
  n = 128
  
  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(x)
  
  x[3] = 123
  x[4] = 456
  assert x[3] == 123
  assert x[4] == 456

test_io()
