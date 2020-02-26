import taichi as ti


@ti.all_archs
def test_compare():
  a = ti.var(ti.i32)
  ti.root.dynamic(ti.i, 256).place(a)
  b = ti.var(ti.i32, shape=())
  c = ti.var(ti.i32, shape=())
  d = ti.var(ti.i32, shape=())

  @ti.kernel
  def func():
    a[2] = 0 <= ti.append(a.parent(), [], 10) < 1
    b[None] = 2
    c[None] = 3
    d[None] = 3
    a[3] = b < c == d
    a[4] = b <= c != d
    a[5] = c > b < d
    a[6] = c == d != b < d > b >= b <= c
    # a[10] = b is not c  # not supported

  func()
  assert a[0] == 10
  assert a[1] == 0  # not appended twice
  assert a[2]
  assert a[3]
  assert not a[4]
  assert a[5]
  assert a[6]