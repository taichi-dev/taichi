import taichi as ti


@ti.all_archs
def test_compare():
  a = ti.var(ti.i32)
  ti.root.dynamic(ti.i, 256).place(a)

  @ti.kernel
  def func():
    # t = 0 <= ti.append(a.parent(), [], 1) < 2
    t = 1 < 0
    print(t)
    # t = ti.chain_compare([0, 1], [2])

  func()
  assert a[1] == 0