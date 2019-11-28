import taichi as ti

@ti.all_archs
def test_3d():
  val = ti.var(ti.i32)

  n = 4
  m = 7
  p = 11

  @ti.layout
  def values():
    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

  assert val.dim() == 3
  print(val.shape())
