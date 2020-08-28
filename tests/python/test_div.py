import taichi as ti


@ti.all_archs
def _test_floor_div(arg1, a, arg2, b, arg3, c):
    z = ti.field(arg3, shape=())

    @ti.kernel
    def func(x: arg1, y: arg2):
        z[None] = x // y

    func(a, b)
    assert z[None] == c


@ti.all_archs
def _test_true_div(arg1, a, arg2, b, arg3, c):
    z = ti.field(arg3, shape=())

    @ti.kernel
    def func(x: arg1, y: arg2):
        z[None] = x / y

    func(a, b)
    assert z[None] == c


def test_floor_div():
    _test_floor_div(ti.i32, 10, ti.i32, 3, ti.f32, 3)
    _test_floor_div(ti.f32, 10, ti.f32, 3, ti.f32, 3)
    _test_floor_div(ti.i32, 10, ti.f32, 3, ti.f32, 3)
    _test_floor_div(ti.f32, 10, ti.i32, 3, ti.f32, 3)

    _test_floor_div(ti.i32, -10, ti.i32, 3, ti.f32, -4)
    _test_floor_div(ti.f32, -10, ti.f32, 3, ti.f32, -4)
    _test_floor_div(ti.i32, -10, ti.f32, 3, ti.f32, -4)
    _test_floor_div(ti.f32, -10, ti.i32, 3, ti.f32, -4)

    _test_floor_div(ti.i32, 10, ti.i32, -3, ti.f32, -4)
    _test_floor_div(ti.f32, 10, ti.f32, -3, ti.f32, -4)
    _test_floor_div(ti.i32, 10, ti.f32, -3, ti.f32, -4)
    _test_floor_div(ti.f32, 10, ti.i32, -3, ti.f32, -4)


def test_true_div():
    _test_true_div(ti.i32, 3, ti.i32, 2, ti.f32, 1.5)
    _test_true_div(ti.f32, 3, ti.f32, 2, ti.f32, 1.5)
    _test_true_div(ti.i32, 3, ti.f32, 2, ti.f32, 1.5)
    _test_true_div(ti.f32, 3, ti.i32, 2, ti.f32, 1.5)
    _test_true_div(ti.f32, 3, ti.i32, 2, ti.i32, 1)

    _test_true_div(ti.i32, -3, ti.i32, 2, ti.f32, -1.5)
    _test_true_div(ti.f32, -3, ti.f32, 2, ti.f32, -1.5)
    _test_true_div(ti.i32, -3, ti.f32, 2, ti.f32, -1.5)
    _test_true_div(ti.f32, -3, ti.i32, 2, ti.f32, -1.5)
    _test_true_div(ti.f32, -3, ti.i32, 2, ti.i32, -1)


@ti.test()
def test_div_default_ip():
    ti.get_runtime().set_default_ip(ti.i64)
    z = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = 1e15 + 1e9
        z[None] = a // 1e10

    func()
    assert z[None] == 100000


@ti.test()
def test_floor_div_pythonic():
    z = ti.field(ti.i32, shape=())

    @ti.kernel
    def func(x: ti.i32, y: ti.i32):
        z[None] = x // y

    for i in range(-10, 11):
        for j in range(-10, 11):
            if j != 0:
                func(i, j)
                assert z[None] == i // j
