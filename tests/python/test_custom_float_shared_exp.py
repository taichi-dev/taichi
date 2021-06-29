import pytest
from pytest import approx

import taichi as ti


@pytest.mark.parametrize('exponent_bits', [5, 6, 7, 8])
@ti.test(require=ti.extension.quant)
def test_shared_exponents(exponent_bits):
    exp = ti.quant.int(exponent_bits, False)
    cit1 = ti.quant.int(10, False)
    cit2 = ti.quant.int(14, False)
    cft1 = ti.type_factory.custom_float(significand_type=cit1,
                                        exponent_type=exp,
                                        scale=1)
    cft2 = ti.type_factory.custom_float(significand_type=cit2,
                                        exponent_type=exp,
                                        scale=1)
    a = ti.field(dtype=cft1)
    b = ti.field(dtype=cft2)
    ti.root.bit_struct(num_bits=32).place(a, b, shared_exponent=True)

    assert a[None] == 0.0
    assert b[None] == 0.0

    a[None] = 10
    assert a[None] == 10.0
    assert b[None] == 0.0

    a[None] = 0
    assert a[None] == 0.0
    assert b[None] == 0.0

    @ti.kernel
    def foo(x: ti.f32, y: ti.f32):
        a[None] = x
        b[None] = y

    foo(3.2, 0.25)

    assert a[None] == approx(3.2, rel=1e-3)
    assert b[None] == approx(0.25, rel=2e-2)
    a[None] = 0.27
    assert a[None] == approx(0.27, rel=1e-2)
    assert b[None] == approx(0.25, rel=2e-2)
    a[None] = 100
    assert a[None] == approx(100, rel=1e-3)
    assert b[None] == approx(0.25, rel=1e-2)

    b[None] = 0
    assert a[None] == approx(100, rel=1e-3)
    assert b[None] == 0

    foo(0, 0)
    assert a[None] == 0.0
    assert b[None] == 0.0

    # test flush to zero
    foo(1000, 1e-6)
    assert a[None] == 1000.0
    assert b[None] == 0.0

    foo(1000, 1000)
    assert a[None] == 1000.0
    assert b[None] == 1000.0

    foo(1e-30, 1e-30)
    if exponent_bits == 8:
        assert a[None] == approx(1e-30, 1e-3)
        assert b[None] == approx(1e-30, 1e-4)
    else:
        # Insufficient exponent bits: should flush to zero
        assert a[None] == 0
        assert b[None] == 0


@pytest.mark.parametrize('exponent_bits', [5, 6, 7, 8])
@ti.test(require=ti.extension.quant)
def test_shared_exponent_add(exponent_bits):
    exp = ti.quant.int(exponent_bits, False)
    cit1 = ti.quant.int(10, False)
    cit2 = ti.quant.int(14, False)
    cft1 = ti.type_factory.custom_float(significand_type=cit1,
                                        exponent_type=exp,
                                        scale=1)
    cft2 = ti.type_factory.custom_float(significand_type=cit2,
                                        exponent_type=exp,
                                        scale=1)
    a = ti.field(dtype=cft1)
    b = ti.field(dtype=cft2)
    ti.root.bit_struct(num_bits=32).place(a, b, shared_exponent=True)

    @ti.kernel
    def foo(x: ti.f32, y: ti.f32):
        a[None] = x
        b[None] = y

    a[None] = 4
    assert a[None] == 4
    assert b[None] == 0
    b[None] = 3
    assert a[None] == 4
    assert b[None] == 3

    b[None] += 1

    assert a[None] == 4
    assert b[None] == 4

    for i in range(100):
        a[None] += 4
        b[None] += 1
        assert a[None] == 4 + (i + 1) * 4
        assert b[None] == 4 + (i + 1)


@pytest.mark.parametrize('exponent_bits', [5, 6, 7, 8])
@ti.test(require=ti.extension.quant)
def test_shared_exponent_borrow(exponent_bits):
    exp = ti.quant.int(exponent_bits, False)
    cit1 = ti.quant.int(10, False)
    cit2 = ti.quant.int(14, False)
    cft1 = ti.type_factory.custom_float(significand_type=cit1,
                                        exponent_type=exp,
                                        scale=1)
    cft2 = ti.type_factory.custom_float(significand_type=cit2,
                                        exponent_type=exp,
                                        scale=1)
    a = ti.field(dtype=cft1)
    b = ti.field(dtype=cft2)
    ti.root.bit_struct(num_bits=32).place(a, b, shared_exponent=True)

    @ti.kernel
    def foo(x: ti.f32, y: ti.f32):
        a[None] = x
        b[None] = y

    def inc():
        a[None] += 1
        b[None] -= 1

    foo(0, 100)

    for i in range(100):
        assert a[None] == i
        assert b[None] == 100 - i
        inc()


@pytest.mark.parametrize('exponent_bits', [5, 6, 7, 8])
@ti.test(require=ti.extension.quant)
def test_negative(exponent_bits):
    exp = ti.quant.int(exponent_bits, False)
    cit1 = ti.quant.int(10, False)
    cit2 = ti.quant.int(14, True)
    cft1 = ti.type_factory.custom_float(significand_type=cit1,
                                        exponent_type=exp,
                                        scale=1)
    cft2 = ti.type_factory.custom_float(significand_type=cit2,
                                        exponent_type=exp,
                                        scale=1)
    a = ti.field(dtype=cft1)
    b = ti.field(dtype=cft2)
    ti.root.bit_struct(num_bits=32).place(a, b, shared_exponent=True)

    a[None] = 37
    assert a[None] == 37
    b[None] = -123
    assert b[None] == -123


# TODO: test precision
# TODO: make sure unsigned has one more effective significand bit
# TODO: test shared exponent floats with custom int in a single bit struct
