import taichi as ti


def _test_pow_f(dt):
    z = ti.field(dt, shape=())

    @ti.kernel
    def func(x: dt, y: dt):
        z[None] = x**y

    for x in [0.5, 1, 1.5, 2, 6.66]:
        for y in [-2, -1, -0.3, 0, 0.5, 1, 1.4, 2.6]:
            func(x, y)
            assert abs(z[None] / x**y - 1) < 0.00001


def _test_pow_i(dt):
    z = ti.field(dt, shape=())

    @ti.kernel
    def func(x: dt, y: ti.template()):
        z[None] = x**y

    for x in range(-5, 5):
        for y in range(0, 4):
            func(x, y)
            assert z[None] == x**y


@ti.all_archs
def test_pow_f32():
    _test_pow_f(ti.f32)


@ti.require(ti.extension.data64)
@ti.all_archs
def test_pow_f64():
    _test_pow_f(ti.f64)


@ti.all_archs
def test_pow_i32():
    _test_pow_i(ti.i32)


@ti.require(ti.extension.data64)
@ti.all_archs
def test_pow_i64():
    _test_pow_i(ti.i64)
