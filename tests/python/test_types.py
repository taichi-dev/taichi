import pytest

import taichi as ti

_TI_TYPES = [ti.i8, ti.i16, ti.i32, ti.u8, ti.u16, ti.u32, ti.f32]
_TI_64_TYPES = [ti.i64, ti.u64, ti.f64]


def _test_type_assign_argument(dt):
    x = ti.field(dt, shape=())

    @ti.kernel
    def func(value: dt):
        x[None] = value

    func(3)
    assert x[None] == 3


@pytest.mark.parametrize('dt', _TI_TYPES)
@ti.archs_excluding(ti.opengl)
def test_type_assign_argument(dt):
    _test_type_assign_argument(dt)


@pytest.mark.parametrize('dt', _TI_64_TYPES)
@ti.require(ti.extension.data64)
@ti.archs_excluding(ti.opengl)
def test_type_assign_argument64(dt):
    _test_type_assign_argument(dt)


def _test_type_operator(dt):
    x = ti.field(dt, shape=())
    y = ti.field(dt, shape=())
    add = ti.field(dt, shape=())
    mul = ti.field(dt, shape=())

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


@pytest.mark.parametrize('dt', _TI_TYPES)
@ti.archs_excluding(ti.opengl)
def test_type_operator(dt):
    _test_type_operator(dt)


@pytest.mark.parametrize('dt', _TI_64_TYPES)
@ti.require(ti.extension.data64)
@ti.archs_excluding(ti.opengl)
def test_type_operator64(dt):
    _test_type_operator(dt)


def _test_type_field(dt):
    x = ti.field(dt, shape=(3, 2))

    @ti.kernel
    def func(i: ti.i32, j: ti.i32):
        x[i, j] = 3

    for i in range(0, 3):
        for j in range(0, 2):
            func(i, j)
            assert x[i, j] == 3


@pytest.mark.parametrize('dt', _TI_TYPES)
@ti.archs_excluding(ti.opengl)
def test_type_field(dt):
    _test_type_field(dt)


@pytest.mark.parametrize('dt', _TI_64_TYPES)
@ti.require(ti.extension.data64)
@ti.archs_excluding(ti.opengl)
def test_type_field64(dt):
    _test_type_field(dt)


def _test_overflow(dt, n):
    a = ti.field(dt, shape=())
    b = ti.field(dt, shape=())
    c = ti.field(dt, shape=())

    @ti.kernel
    def func():
        c[None] = a[None] + b[None]

    a[None] = 2**n // 3
    b[None] = 2**n // 3

    func()

    assert a[None] == 2**n // 3
    assert b[None] == 2**n // 3

    if ti.core.is_signed(dt):
        assert c[None] == 2**n // 3 * 2 - (2**n)  # overflows
    else:
        assert c[None] == 2**n // 3 * 2  # does not overflow


@pytest.mark.parametrize('dt,n', [
    (ti.i8, 8),
    (ti.u8, 8),
    (ti.i16, 16),
    (ti.u16, 16),
    (ti.i32, 32),
    (ti.u32, 32),
])
@ti.archs_excluding(ti.opengl)
def test_overflow(dt, n):
    _test_overflow(dt, n)


@pytest.mark.parametrize('dt,n', [
    (ti.i64, 64),
    (ti.u64, 64),
])
@ti.require(ti.extension.data64)
@ti.archs_excluding(ti.opengl)
def test_overflow64(dt, n):
    _test_overflow(dt, n)


@pytest.mark.parametrize('dt,val', [
    (ti.u32, 0xffffffff),
    (ti.u64, 0xffffffffffffffff),
])
@ti.test(require=ti.extension.data64)
def test_uint_max(dt, val):
    # https://github.com/taichi-dev/taichi/issues/2060
    ti.get_runtime().default_ip = dt
    N = 16
    f = ti.field(dt, shape=N)

    @ti.kernel
    def run():
        for i in f:
            f[i] = val

    run()
    fs = f.to_numpy()
    for f in fs:
        assert f == val
