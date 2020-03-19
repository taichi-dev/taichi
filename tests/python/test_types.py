import taichi as ti
import pytest

_TI_TYPES = [ti.i8, ti.i16, ti.i32, ti.u8, ti.u16, ti.u32, ti.f32]
_TI_64_TYPES = [ti.i64, ti.u64, ti.f64]


def _test_type_assign_argument(dt):
    x = ti.var(dt, shape=())

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
    x = ti.var(dt, shape=())
    y = ti.var(dt, shape=())
    add = ti.var(dt, shape=())
    mul = ti.var(dt, shape=())

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


def _test_type_tensor(dt):
    x = ti.var(dt, shape=(3, 2))

    @ti.kernel
    def func(i: ti.i32, j: ti.i32):
        x[i, j] = 3

    for i in range(0, 3):
        for j in range(0, 2):
            func(i, j)
            assert x[i, j] == 3


@pytest.mark.parametrize('dt', _TI_TYPES)
@ti.archs_excluding(ti.opengl)
def test_type_tensor(dt):
    _test_type_tensor(dt)


@pytest.mark.parametrize('dt', _TI_64_TYPES)
@ti.require(ti.extension.data64)
@ti.archs_excluding(ti.opengl)
def test_type_tensor64(dt):
    _test_type_tensor(dt)


def _test_overflow(dt, n):
    a = ti.var(dt, shape=())
    b = ti.var(dt, shape=())
    c = ti.var(dt, shape=())

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
