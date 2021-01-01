import taichi as ti
import numpy as np
import pytest


@ti.test(require=ti.extension.quant)
def test_custom_float_unsigned():
    cu13 = ti.type_factory.custom_int(13, False)
    exp = ti.type_factory.custom_int(6, False)
    cft = ti.type_factory.custom_float(significand_type=cu13,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    tests = [
        0, 1 / 1024, 1.75 / 1024, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 128,
        256, 512, 1024
    ]

    assert x[None] == 0

    for v in tests:
        x[None] = v
        assert x[None] == v


@ti.test(require=ti.extension.quant)
def test_custom_float_signed():
    cu13 = ti.type_factory.custom_int(13, True)
    exp = ti.type_factory.custom_int(6, False)
    cft = ti.type_factory.custom_float(significand_type=cu13,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    tests = [0, 0.125, 0.5, 2, 4, 6, 7, 8, 9]

    assert x[None] == 0

    for v in tests:
        x[None] = v
        print(x[None])
        assert x[None] == v

        x[None] = -v
        assert x[None] == -v


@pytest.mark.parametrize('digits_bits', [23, 24])
@ti.test(require=ti.extension.quant)
def test_custom_float_precision(digits_bits):
    cu24 = ti.type_factory.custom_int(digits_bits, True)
    exp = ti.type_factory.custom_int(8, False)
    cft = ti.type_factory.custom_float(significand_type=cu24,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    tests = [np.float32(np.pi), np.float32(np.pi * (1 << 100))]

    for v in tests:
        x[None] = v
        if digits_bits == 24:
            # Sufficient digits
            assert x[None] == v
        else:
            # The binary representation of np.float32(np.pi) ends with 1, so removing one digit will result in a different number.
            assert x[None] != v
            assert x[None] == pytest.approx(v, rel=3e-7)


@pytest.mark.parametrize('signed', [True, False])
@ti.test(require=ti.extension.quant)
def test_custom_float_truncation(signed):
    cit = ti.type_factory.custom_int(2, signed)
    exp = ti.type_factory.custom_int(5, False)
    cft = ti.type_factory.custom_float(significand_type=cit,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    # Sufficient digits
    for v in [1, 1.5]:
        x[None] = v
        assert x[None] == v

    x[None] = 1.75
    if signed:
        # Insufficient digits
        assert x[None] == 1.5
    else:
        # Sufficient digits
        assert x[None] == 1.75

    # Insufficient digits
    x[None] = 1.875
    if signed:
        assert x[None] == 1.5
    else:
        assert x[None] == 1.75
