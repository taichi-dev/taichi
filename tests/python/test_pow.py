import pytest
from taichi.lang.exception import TaichiRuntimeError

import taichi as ti
from tests import test_utils


def _test_pow_f(dt):
    z = ti.field(dt, shape=())

    @ti.kernel
    def func(x: dt, y: dt):
        z[None] = x**y

    for x in [0.5, 1, 1.5, 2, 6.66]:
        for y in [-2, -1, -0.3, 0, 0.5, 1, 1.4, 2.6]:
            func(x, y)
            assert z[None] == pytest.approx(x**y)


def _test_pow_i(dt):
    z = ti.field(dt, shape=())

    @ti.kernel
    def func(x: dt, y: ti.template()):
        z[None] = x**y

    for x in range(-5, 5):
        for y in range(0, 4):
            func(x, y)
            assert z[None] == x**y


@test_utils.test()
def test_pow_f32():
    _test_pow_f(ti.f32)


@test_utils.test(require=ti.extension.data64)
def test_pow_f64():
    _test_pow_f(ti.f64)


@test_utils.test()
def test_pow_i32():
    _test_pow_i(ti.i32)


@test_utils.test(require=ti.extension.data64)
def test_pow_i64():
    _test_pow_i(ti.i64)


def _ipow_negative_exp(dt):
    z = ti.field(dt, shape=())

    @ti.kernel
    def foo(x: dt, y: ti.template()):
        z[None] = x**y

    with pytest.raises(TaichiRuntimeError):
        foo(10, -10)


@test_utils.test(
    debug=True,
    advanced_optimization=False,
    exclude=[ti.vulkan, ti.metal, ti.opengl, ti.gles],
)
def test_ipow_negative_exp_i32():
    _ipow_negative_exp(ti.i32)


@test_utils.test(
    debug=True,
    advanced_optimization=False,
    require=ti.extension.data64,
    exclude=[ti.vulkan, ti.metal, ti.opengl, ti.gles],
)
def test_ipow_negative_exp_i64():
    _ipow_negative_exp(ti.i64)


def _test_pow_int_base_int_exp(dt_base, dt_exp):
    z = ti.field(dt_base, shape=())

    @ti.kernel
    def func(x: dt_base, y: dt_exp):
        z[None] = x**y

    for x in range(-5, 5):
        for y in range(0, 10):
            func(x, y)
            assert z[None] == x**y


@test_utils.test()
def test_pow_int_base_int_exp_32():
    _test_pow_int_base_int_exp(ti.i32, ti.i32)


@pytest.mark.parametrize("dt_base, dt_exp", [(ti.i32, ti.i64), (ti.i64, ti.i64), (ti.i64, ti.i32)])
@test_utils.test(require=ti.extension.data64)
def test_pow_int_base_int_exp_64(dt_base, dt_exp):
    _test_pow_int_base_int_exp(dt_base, dt_exp)


def _test_pow_float_base_int_exp(dt_base, dt_exp):
    z = ti.field(dt_base, shape=())

    @ti.kernel
    def func(x: dt_base, y: dt_exp):
        z[None] = x**y

    for x in [-6.66, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 6.66]:
        for y in range(-10, 10):
            func(x, y)
            assert z[None] == pytest.approx(x**y)


@test_utils.test()
def test_pow_float_base_int_exp_32():
    _test_pow_float_base_int_exp(ti.f32, ti.i32)


@pytest.mark.parametrize("dt_base, dt_exp", [(ti.f64, ti.i32), (ti.f32, ti.i64), (ti.f64, ti.i64)])
@test_utils.test(require=ti.extension.data64)
def test_pow_float_base_int_exp_64(dt_base, dt_exp):
    _test_pow_float_base_int_exp(dt_base, dt_exp)
