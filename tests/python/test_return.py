import taichi as ti
from taichi import approx


@ti.must_throw(ti.TaichiSyntaxError)
def _test_return_not_last_stmt():  # TODO: make this work
    x = ti.field(ti.i32, ())

    @ti.kernel
    def kernel() -> ti.i32:
        return 1
        x[None] = 233

    kernel()


@ti.must_throw(ti.TaichiSyntaxError)
def test_return_without_type_hint():
    @ti.kernel
    def kernel():
        return 1

    kernel()


def test_const_func_ret():
    @ti.kernel
    def func1() -> ti.f32:
        return 3

    @ti.kernel
    def func2() -> ti.i32:
        return 3.3  # return type mismatch, will be auto-casted into ti.i32

    assert func1() == approx(3)
    assert func2() == 3


@ti.all_archs
def _test_binary_func_ret(dt1, dt2, dt3, castor):
    @ti.kernel
    def func(a: dt1, b: dt2) -> dt3:
        return a * b

    if ti.core.is_integral(dt1):
        xs = list(range(4))
    else:
        xs = [0.2, 0.4, 0.8, 1.0]

    if ti.core.is_integral(dt2):
        ys = list(range(4))
    else:
        ys = [0.2, 0.4, 0.8, 1.0]

    for x, y in zip(xs, ys):
        assert func(x, y) == approx(castor(x * y))


def test_binary_func_ret():
    _test_binary_func_ret(ti.i32, ti.f32, ti.f32, float)
    _test_binary_func_ret(ti.f32, ti.i32, ti.f32, float)
    _test_binary_func_ret(ti.i32, ti.f32, ti.i32, int)
    _test_binary_func_ret(ti.f32, ti.i32, ti.i32, int)
