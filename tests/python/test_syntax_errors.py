import taichi as ti

@ti.must_throw(ti.TaichiSyntaxError)
def test_try():
  ti.get_runtime().print_preprocessed = True
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
