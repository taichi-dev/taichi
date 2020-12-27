import taichi as ti
import numpy as np
import pytest


@ti.test(require=ti.extension.quant)
def test_custom_float_unsigned():
    cu13 = ti.type_factory_.get_custom_int_type(13, False)
    exp = ti.type_factory_.get_custom_int_type(6, False)
    cft = ti.type_factory.custom_float(significand_type=cu13,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    tests = [
        1 / 1024, 1.75 / 1024, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 128, 256,
        512, 1024
    ]

    for v in tests:
        x[None] = v
        assert x[None] == v


@ti.test(require=ti.extension.quant)
def test_custom_float_signed():
    cu13 = ti.type_factory_.get_custom_int_type(13, True)
    exp = ti.type_factory_.get_custom_int_type(6, False)
    cft = ti.type_factory.custom_float(significand_type=cu13,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    tests = [-2, -4, -6, -7, -8, -9]

    for v in tests:
        x[None] = v
        assert x[None] == v


@pytest.mark.parametrize('digits_bits', [23, 24])
@ti.test(require=ti.extension.quant)
def test_custom_float_precision(digits_bits):
    cu24 = ti.type_factory_.get_custom_int_type(digits_bits, True)
    exp = ti.type_factory_.get_custom_int_type(8, False)
    cft = ti.type_factory.custom_float(significand_type=cu24,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    tests = [np.float32(np.pi), np.float32(np.pi * 1e38)]

    for v in tests:
        x[None] = v
        if digits_bits == 24:
            assert x[None] == v
        else:
            # Insufficient digits
            assert x[None] != v
            assert x[None] == pytest.approx(v)


@ti.test(require=ti.extension.quant)
def test_custom_float_truncation():
    return
    cu24 = ti.type_factory_.get_custom_int_type(3, True)
    exp = ti.type_factory_.get_custom_int_type(8, False)
    cft = ti.type_factory.custom_float(significand_type=cu24,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    tests = [np.float32(np.pi)]

    for v in tests:
        x[None] = v
        if digits_bits == 24:
            assert x[None] == v
        else:
            # Insufficient digits
            assert x[None] != v
            assert x[None] == pytest.approx(v)
