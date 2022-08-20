import numpy as np
import pytest
from pytest import approx

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize('max_num_bits', [32, 64])
@test_utils.test(require=ti.extension.quant)
def test_quant_float_unsigned(max_num_bits):
    qflt = ti.types.quant.float(exp=6, frac=13, signed=False)
    x = ti.field(dtype=qflt)

    bitpack = ti.BitpackedFields(max_num_bits=max_num_bits)
    bitpack.place(x)
    ti.root.place(bitpack)

    tests = [
        0, 1 / 1024, 1.75 / 1024, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 128,
        256, 512, 1024
    ]

    assert x[None] == 0

    for v in tests:
        x[None] = v
        assert x[None] == v


@test_utils.test(require=ti.extension.quant)
def test_quant_float_signed():
    qflt = ti.types.quant.float(exp=6, frac=13, signed=True)
    x = ti.field(dtype=qflt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

    tests = [0, 0.125, 0.5, 2, 4, 6, 7, 8, 9]

    assert x[None] == 0

    for v in tests:
        x[None] = v
        assert x[None] == v

        x[None] = -v
        assert x[None] == -v

    ftz_tests = [1e-30, 1e-20, 1e-10, 1e-2]
    for v in ftz_tests:
        x[None] = v
        assert x[None] == approx(v, abs=1e-5)

        x[None] = -v
        assert x[None] == approx(-v, abs=1e-5)


@pytest.mark.parametrize('digits_bits', [23, 24])
@test_utils.test(require=ti.extension.quant)
def test_quant_float_precision(digits_bits):
    qflt = ti.types.quant.float(exp=8, frac=digits_bits)
    x = ti.field(dtype=qflt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

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
@test_utils.test(require=ti.extension.quant)
def test_quant_float_truncation(signed):
    qflt = ti.types.quant.float(exp=5, frac=2, signed=signed)
    x = ti.field(dtype=qflt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

    # Sufficient digits
    for v in [1, 1.5]:
        x[None] = v
        assert x[None] == v

    x[None] = 1.75
    if signed:
        # Insufficient digits
        assert x[None] == 2
    else:
        # Sufficient digits
        assert x[None] == 1.75

    # Insufficient digits
    x[None] = 1.625
    if signed:
        assert x[None] == 1.5
    else:
        assert x[None] == 1.75


@test_utils.test(require=ti.extension.quant)
def test_quant_float_atomic_demotion():
    qflt = ti.types.quant.float(exp=5, frac=2)
    x = ti.field(dtype=qflt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

    @ti.kernel
    def foo():
        for i in range(1):
            x[None] += 1

    foo()
    foo()

    assert x[None] == 2
