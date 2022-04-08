import pytest
from taichi.lang import impl

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize("arg1,a,arg2,b,arg3,c", [
    (ti.i32, 10, ti.i32, 3, ti.f32, 3),
    (ti.f32, 10, ti.f32, 3, ti.f32, 3),
    (ti.i32, 10, ti.f32, 3, ti.f32, 3),
    (ti.f32, 10, ti.i32, 3, ti.f32, 3),
    (ti.i32, -10, ti.i32, 3, ti.f32, -4),
    (ti.f32, -10, ti.f32, 3, ti.f32, -4),
    (ti.i32, -10, ti.f32, 3, ti.f32, -4),
    (ti.f32, -10, ti.i32, 3, ti.f32, -4),
    (ti.i32, 10, ti.i32, -3, ti.f32, -4),
    (ti.f32, 10, ti.f32, -3, ti.f32, -4),
    (ti.i32, 10, ti.f32, -3, ti.f32, -4),
    (ti.f32, 10, ti.i32, -3, ti.f32, -4),
])
@test_utils.test()
def test_floor_div(arg1, a, arg2, b, arg3, c):
    z = ti.field(arg3, shape=())

    @ti.kernel
    def func(x: arg1, y: arg2):
        z[None] = x // y

    func(a, b)
    assert z[None] == c


@pytest.mark.parametrize("arg1,a,arg2,b,arg3,c", [
    (ti.i32, 3, ti.i32, 2, ti.f32, 1.5),
    (ti.f32, 3, ti.f32, 2, ti.f32, 1.5),
    (ti.i32, 3, ti.f32, 2, ti.f32, 1.5),
    (ti.f32, 3, ti.i32, 2, ti.f32, 1.5),
    (ti.f32, 3, ti.i32, 2, ti.i32, 1),
    (ti.i32, -3, ti.i32, 2, ti.f32, -1.5),
    (ti.f32, -3, ti.f32, 2, ti.f32, -1.5),
    (ti.i32, -3, ti.f32, 2, ti.f32, -1.5),
    (ti.f32, -3, ti.i32, 2, ti.f32, -1.5),
    (ti.f32, -3, ti.i32, 2, ti.i32, -1),
])
@test_utils.test()
def test_true_div(arg1, a, arg2, b, arg3, c):
    z = ti.field(arg3, shape=())

    @ti.kernel
    def func(x: arg1, y: arg2):
        z[None] = x / y

    func(a, b)
    assert z[None] == c


@test_utils.test()
def test_div_default_ip():
    impl.get_runtime().set_default_ip(ti.i64)
    z = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = 1e15 + 1e9
        z[None] = a // 1e10

    func()
    assert z[None] == 100000


@test_utils.test()
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
