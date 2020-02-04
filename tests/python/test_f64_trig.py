import taichi as ti


@ti.all_archs
def test_f64_sin():
  val = ti.var(ti.f64)

  @ti.layout
  def values():
    ti.root.dense(ti.i, 8).place(val)

  @ti.kernel
  def test_sin():
    for i in range(8):
      val[i] = ti.sin(ti.cast(i,ti.f64))

  test_sin()

@ti.all_archs
def test_f64_cos():
  val = ti.var(ti.f64)

  @ti.layout
  def values():
    ti.root.dense(ti.i, 8).place(val)

  @ti.kernel
  def test_cos():
    for i in range(8):
      val[i] = ti.cos(ti.cast(i,ti.f64))

  test_cos()
