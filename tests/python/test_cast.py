import taichi as ti
import pytest


@ti.all_archs
def test_cast_f32():
    z = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        z[None] = ti.cast(1e9, ti.f32) / ti.cast(1e6, ti.f32) + 1e-3

    func()
    assert z[None] == 1000


@ti.require(ti.extension.data64)
@ti.all_archs
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


@ti.all_archs
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


@ti.all_archs
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


@ti.test(ti.cpu, ti.cuda, cfg_optimization=False)
def test_int_extension():
    x = ti.field(dtype=ti.i32, shape=1)
    y = ti.field(dtype=ti.u16, shape=1)

    @ti.kernel
    def run_cast():
        x[0] = ti.cast(y[0], ti.i32)

    y[0] = 0xFFFF
    run_cast()
    assert x[0] == 0xFFFF

    y[0] = 0x8234
    run_cast()
    assert x[0] == 0x8234


# TODO: add arch=ti.cuda after bit-pointer is supported on cuda
@ti.test(ti.cpu, cfg_optimization=False)
def test_custom_int_extension():
    x = ti.field(dtype=ti.i32, shape=2)

    ci5 = ti.type_factory_.get_custom_int_type(5, True, 16)
    cu7 = ti.type_factory_.get_custom_int_type(7, False, 16)

    a = ti.field(dtype=ci5)
    b = ti.field(dtype=cu7)

    ti.root._bit_struct(num_bits=32).place(a, b)

    @ti.kernel
    def run_cast():
        x[0] = ti.cast(a[None], ti.i32)
        x[1] = ti.cast(b[None], ti.i32)

    a[None] = 0x1F
    b[None] = 0x3F
    run_cast()
    assert x[0] == -1
    assert x[1] == 0x3F
