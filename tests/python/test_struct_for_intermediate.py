import taichi as ti

@ti.all_archs
def test_nested():
  return
  x = ti.var(ti.i32)

  n = 2

  @ti.layout
  def place():
    ti.root.dense(ti.ij, n).dense(ti.ij, n).place(x)

  @ti.kernel
  def iterate():
    for i, j in x.parent():
      print(i)
      print(j)
      x[i, j] = i + j * 2

  iterate()

test_nested()
