import taichi as ti

def all_data_types_and_test(foo):
  def wrapped():
    tests = []
    for dt in [ti.i32, ti.i64, ti.i8, ti.i16, ti.u8, ti.u16, ti.u32, ti.u64, ti.f32, ti.f64]:
      tests.append(foo(dt))
    for test in tests:
      # variables are expected to be declared before kernel invocation, discuss at:
      # https://github.com/taichi-dev/taichi/pull/505#issuecomment-588644274
      test()
  return wrapped

@ti.all_archs
@all_data_types_and_test
def test_type_assign_argument(dt):
  x = ti.var(dt, shape=())

  def tester():
    @ti.kernel
    def func(value: dt):
      x[None] = value

    func(3)
    assert x[None] == 3

  return tester

@ti.all_archs
@all_data_types_and_test
def test_type_operator(dt):
  x = ti.var(dt, shape=())
  y = ti.var(dt, shape=())
  add = ti.var(dt, shape=())
  mul = ti.var(dt, shape=())

  def tester():
    @ti.kernel
    def func():
      add[None] = x[None] + y[None]
      mul[None] = x[None] * y[None]

    for i in range(0, 3):
      for j in range(0, 3):
        x[None] = i
        y[None] = j
        func()
        assert add[None] == x[None] + y[None]
        assert mul[None] == x[None] * y[None]

  return tester

@ti.all_archs
@all_data_types_and_test
def test_type_tensor(dt):
  x = ti.var(dt, shape=(3, 2))

  def tester():
    @ti.kernel
    def func(i: ti.i32, j: ti.i32):
      x[i, j] = 3

    for i in range(0, 3):
      for j in range(0, 2):
        func(i, j)
        assert x[i, j] == 3

  return tester

@ti.all_archs
def _test_overflow(dt, n):
  a = ti.var(dt, shape=())
  b = ti.var(dt, shape=())
  c = ti.var(dt, shape=())

  @ti.kernel
  def func():
    c[None] = a[None] + b[None]

  a[None] = 2 ** n // 3
  b[None] = 2 ** n // 3

  func()

  assert a[None] == 2 ** n // 3
  assert b[None] == 2 ** n // 3

  if ti.core.is_signed(dt):
    assert c[None] == 2 ** n // 3 * 2 - (2 ** n) # overflows
  else:
    assert c[None] == 2 ** n // 3 * 2 # does not overflow

def test_overflow():
  _test_overflow(ti.i8, 8)
  _test_overflow(ti.u8, 8)
  _test_overflow(ti.i16, 16)
  _test_overflow(ti.u16, 16)
  _test_overflow(ti.i32, 32)
  _test_overflow(ti.u32, 32)
  _test_overflow(ti.i64, 64)
  _test_overflow(ti.u64, 64)
