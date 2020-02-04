import taichi as ti


@ti.all_archs
def test_f64_sin():
  from numpy import sin
  val = ti.var(ti.f64)

  @ti.layout
  def values():
    ti.root.dense(ti.i, 8).place(val)

  @ti.kernel
  def fill_sin():
    for i in range(8):
      val[i] = ti.sin(ti.cast(i,ti.f64))

  fill_sin()

  # check that it is double precision
  for i in range(8):
    assert(abs(sin(float(i))-val[i]) < 1.0e-15)

@ti.all_archs
def test_f64_cos():
  from numpy import cos
  val = ti.var(ti.f64)

  @ti.layout
  def values():
    ti.root.dense(ti.i, 8).place(val)

  @ti.kernel
  def fill_cos():
    for i in range(8):
      val[i] = ti.cos(ti.cast(i,ti.f64))

  fill_cos()

  # check that it is double precision
  for i in range(8):
    assert(abs(cos(float(i))-val[i]) < 1.0e-15)
