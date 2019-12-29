import taichi as ti

# @ti.all_archs
def test_dynamic():
  x = ti.var(ti.i32)
  y = ti.var(ti.i32, shape=())
  ti.cfg.print_ir = True

  n = 128

  @ti.layout
  def place():
    ti.root.dynamic(ti.i, n).place(x)

  @ti.kernel
  def count():
    for i in x:
      y[None] += 1
      
  x[n // 3] = 1

  count()

  print(y[None])
  assert y[None] == n // 3 + 1

test_dynamic()
