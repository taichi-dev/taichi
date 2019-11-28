import taichi as ti

@ti.all_archs
def test_non_POT():
  x = ti.var(ti.i32)
  sum = ti.var(ti.i32)

  n = 65

  @ti.layout
  def place():
    ti.root.dense(ti.k, n).place(x)
    ti.root.place(sum)

  @ti.kernel
  def accumulate():
    for i in x:
      print(i)
      ti.atomic_add(sum, i)

  accumulate()

  for i in range(n):
    assert sum[None] == 4950
