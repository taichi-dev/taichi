import taichi as ti


@ti.all_archs
def test_vector_index():
  val = ti.var(ti.i32)

  n = 4
  m = 7
  p = 11

  @ti.layout
  def values():
    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

  @ti.kernel
  def test():
    for i in range(n):
      for j in range(m):
        for k in range(p):
          I = ti.Vector([i, j, k])
          val[I] = i + j * 2 + k * 3

  test()

  for i in range(n):
    for j in range(m):
      for k in range(p):
        assert val[i, j, k] == i + j * 2 + k * 3


@ti.all_archs
def test_grouped():
  ti.get_runtime().print_preprocessed = True
  val = ti.var(ti.i32)

  n = 4
  m = 8
  p = 16

  @ti.layout
  def values():
    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

  @ti.kernel
  def test():
    for I in ti.grouped(val):
      val[I] = I[0] + I[1] * 2 + I[2] * 3

  test()

  for i in range(n):
    for j in range(m):
      for k in range(p):
        assert val[i, j, k] == i + j * 2 + k * 3
