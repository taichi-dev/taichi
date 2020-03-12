import taichi as ti

@ti.all_archs
def test_ptr_scalar():
  a = ti.var(dt=ti.f32, shape=())

  @ti.kernel
  def func(t: ti.f32):
    b = ti.static(a)
    c = ti.static(b)
    b[None] = b[None] * t
    c[None] = a[None] + t

  for x, y in zip(range(-5, 5), range(-4, 4)):
    a[None] = x
    func(y)
    assert a[None] == x * y + y

@ti.all_archs
def test_ptr_matrix():
  a = ti.Matrix(2, 2, dt=ti.f32, shape=())

  @ti.kernel
  def func(t: ti.f32):
    a[None] = [[2, 3], [4, 5]]
    b = ti.static(a)
    b[None][1, 0] = t

  for x in range(-5, 5):
    func(x)
    assert a[None][1, 0] == x

@ti.all_archs
def test_ptr_tensor():
  a = ti.var(dt=ti.f32, shape=(3, 4))

  @ti.kernel
  def func(t: ti.f32):
    b = ti.static(a)
    b[1, 3] = b[1, 2] * t
    b[2, 0] = b[2, 1] + t

  for x, y in zip(range(-5, 5), range(-4, 4)):
    a[1, 2] = x
    a[2, 1] = x
    func(y)
    assert a[1, 3] == x * y
    assert a[2, 0] == x + y
