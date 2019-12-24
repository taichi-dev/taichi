import taichi as ti


@ti.all_archs
def test_bitmasked():
  if ti.get_os_name() == 'win':
    # This test not supported on Windows due to the VirtualAlloc issue #251
    return
  x = ti.var(ti.f32)
  s = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).bitmasked().dense(ti.i, n).place(x)
    ti.root.place(s)

  @ti.kernel
  def func():
    for i in x:
      s[None] += 1

  x[0] = 1
  x[127] = 1
  x[256] = 1
  x[257] = 1

  func()
  assert s[None] == 256


@ti.all_archs
def test_pointer():
  if ti.get_os_name() == 'win':
    # This test not supported on Windows due to the VirtualAlloc issue #251
    return
  x = ti.var(ti.f32)
  s = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).pointer().dense(ti.i, n).place(x)
    ti.root.place(s)

  @ti.kernel
  def func():
    for i in x:
      ti.atomic_add(s[None], 1)

  x[0] = 1
  x[127] = 1
  x[256] = 1

  func()
  assert s[None] == 256


@ti.all_archs
def test_pointer2():
  if ti.get_os_name() == 'win':
    # This test not supported on Windows due to the VirtualAlloc issue #251
    return
  x = ti.var(ti.f32)
  s = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).pointer().dense(ti.i, n).pointer().dense(ti.i,
                                                                    n).place(x)
    ti.root.place(s)

  @ti.kernel
  def func():
    for i in x:
      s[None] += 1

  x[0] = 1
  x[127] = 1
  x[254] = 1
  x[256 + n * n] = 1

  x[257 + n * n] = 1
  x[257 + n * n * 2] = 1
  x[257 + n * n * 5] = 1

  func()
  assert s[None] == 5 * n
  print(x[257 + n * n * 7])
  assert s[None] == 5 * n
