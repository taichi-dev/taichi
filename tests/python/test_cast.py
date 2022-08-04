import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize('dtype', [ti.u8, ti.u16, ti.u32])
@test_utils.test(exclude=ti.opengl)
def test_cast_uint_to_float(dtype):
    @ti.kernel
    def func(a: dtype) -> ti.f32:
        return ti.cast(a, ti.f32)

    @ti.kernel
    def func_sugar(a: dtype) -> ti.f32:
        return ti.f32(a)

    assert func(255) == func_sugar(255) == 255


@pytest.mark.parametrize('dtype', [ti.u8, ti.u16, ti.u32])
@test_utils.test(exclude=ti.opengl)
def test_cast_float_to_uint(dtype):
    @ti.kernel
    def func(a: ti.f32) -> dtype:
        return ti.cast(a, dtype)

    @ti.kernel
    def func_sugar(a: ti.f32) -> dtype:
        return dtype(a)

    assert func(255) == func_sugar(255) == 255


@test_utils.test()
def test_cast_f32():
    z = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        z[None] = ti.cast(1e9, ti.f32) / ti.cast(1e6, ti.f32) + 1e-3

    func()
    assert z[None] == 1000


@test_utils.test(require=ti.extension.data64)
def test_cast_f64():
    z = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        z[None] = ti.cast(1e13, ti.f64) / ti.cast(1e10, ti.f64) + 1e-3

    func()
    assert z[None] == 1000


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
def test_cast_default_fp(dtype):
    ti.init(default_fp=dtype)

    @ti.kernel
    def func(x: int, y: int) -> float:
        return ti.cast(x, float) * float(y)

    assert func(23, 4) == pytest.approx(23.0 * 4.0)


@pytest.mark.parametrize('dtype', [ti.i32, ti.i64])
def test_cast_default_ip(dtype):
    ti.init(default_ip=dtype)

    @ti.kernel
    def func(x: float, y: float) -> int:
        return ti.cast(x, int) * int(y)

    # make sure that int(4.6) == 4:
    assert func(23.3, 4.6) == 23 * 4
    if dtype == ti.i64:
        large = 1000000000
        assert func(large, 233) == large * 233
        assert func(233, large) == 233 * large


@test_utils.test()
def test_cast_within_while():
    ret = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        t = 10
        while t > 5:
            t = 1.0
            break
        ret[None] = t

    func()


@test_utils.test()
def test_bit_cast():
    x = ti.field(ti.i32, shape=())
    y = ti.field(ti.f32, shape=())
    z = ti.field(ti.i32, shape=())

    @ti.kernel
    def func1():
        y[None] = ti.bit_cast(x[None], ti.f32)

    @ti.kernel
    def func2():
        z[None] = ti.bit_cast(y[None], ti.i32)

    x[None] = 2333
    func1()
    func2()
    assert z[None] == 2333


@test_utils.test(arch=ti.cpu)
def test_int_extension():
    x = ti.field(dtype=ti.i32, shape=2)
    y = ti.field(dtype=ti.u32, shape=2)

    a = ti.field(dtype=ti.i8, shape=1)
    b = ti.field(dtype=ti.u8, shape=1)

    @ti.kernel
    def run_cast_i32():
        x[0] = ti.cast(a[0], ti.i32)
        x[1] = ti.cast(b[0], ti.i32)

    @ti.kernel
    def run_cast_u32():
        y[0] = ti.cast(a[0], ti.u32)
        y[1] = ti.cast(b[0], ti.u32)

    a[0] = -128
    b[0] = -128

    run_cast_i32()
    assert x[0] == -128
    assert x[1] == 128

    run_cast_u32()
    assert y[0] == 0xFFFFFF80
    assert y[1] == 128


@test_utils.test(arch=ti.cpu)
def test_quant_int_extension():
    x = ti.field(dtype=ti.i32, shape=2)
    y = ti.field(dtype=ti.u32, shape=2)

    qi5 = ti.types.quant.int(5, True, ti.i16)
    qu7 = ti.types.quant.int(7, False, ti.u16)

    a = ti.field(dtype=qi5)
    b = ti.field(dtype=qu7)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(a, b)
    ti.root.place(bitpack)

    @ti.kernel
    def run_cast_int():
        x[0] = ti.cast(a[None], ti.i32)
        x[1] = ti.cast(b[None], ti.i32)

    @ti.kernel
    def run_cast_uint():
        y[0] = ti.cast(a[None], ti.u32)
        y[1] = ti.cast(b[None], ti.u32)

    a[None] = -16
    b[None] = -64

    run_cast_int()
    assert x[0] == -16
    assert x[1] == 64

    run_cast_uint()
    assert y[0] == 0xFFFFFFF0
    assert y[1] == 64
