import taichi as ti

@ti.must_throw(ti.TaichiSyntaxError)
def test_try():
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    try:
      a = 0
    except:
      a = 1

  func()

@ti.must_throw(ti.TaichiSyntaxError)
def test_import():
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    import something

  func()

@ti.must_throw(ti.TaichiSyntaxError)
def test_for_else():
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    for i in range(10):
      pass
    else:
      pass

  func()

@ti.must_throw(ti.TaichiSyntaxError)
def test_while_else():
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    while True:
      pass
    else:
      pass

  func()

@ti.must_throw(ti.TaichiSyntaxError)
def test_break():
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    while True:
      break

  func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_continue():
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    while True:
      continue

  func()
