import taichi as ti

@ti.must_throw(UnboundLocalError)
def test_if():
  ti.get_runtime().print_preprocessed = True
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    if True:
      a = 0
    else:
      a = 1
    ti.print(a)

  func()

@ti.must_throw(UnboundLocalError)
def test_for():
  ti.get_runtime().print_preprocessed = True
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    for i in range(10):
      a = i
    ti.print(a)

  func()

@ti.must_throw(UnboundLocalError)
def test_while():
  ti.get_runtime().print_preprocessed = True
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    while True:
      a = 0
    ti.print(a)

  func()

