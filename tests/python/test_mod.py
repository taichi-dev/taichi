import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize("a,b", [
    (10, 3),
    (-10, 3),
    (10, -3),
    (-10, -3),
])
@test_utils.test()
def test_py_style_mod(a, b):
    z = ti.field(ti.i32, shape=())

    @ti.kernel
    def func(x: ti.i32, y: ti.i32):
        z[None] = x % y

    func(a, b)
    assert z[None] == a % b


@pytest.mark.parametrize("a,b", [
    (10, 3),
    (-10, 3),
    (10, -3),
    (-10, -3),
])
@test_utils.test()
def test_c_style_mod(a, b):
    z = ti.field(ti.i32, shape=())

    @ti.kernel
    def func(x: ti.i32, y: ti.i32):
        z[None] = ti.raw_mod(x, y)

    func(a, b)
    assert z[None] == _c_mod(a, b)


def _c_mod(a, b):
    return a - b * int(float(a) / b)


@test_utils.test()
def test_mod_scan():
    z = ti.field(ti.i32, shape=())
    w = ti.field(ti.i32, shape=())

    @ti.kernel
    def func(x: ti.i32, y: ti.i32):
        z[None] = x % y
        w[None] = ti.raw_mod(x, y)

    for i in range(-10, 11):
        for j in range(-10, 11):
            if j != 0:
                func(i, j)
                assert z[None] == i % j
                assert w[None] == _c_mod(i, j)


@test_utils.test()
def test_py_style_float_const_mod_one():
    @ti.kernel
    def func() -> ti.f32:
        a = 0.5
        return a % 1

    assert func() == 0.5
