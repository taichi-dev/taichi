import math

from pytest import approx

import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_fixed():
    qfxt = ti.types.quant.fixed(bits=32, max_value=2)
    x = ti.field(dtype=qfxt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

    @ti.kernel
    def foo():
        x[None] = 0.7
        print(x[None])
        x[None] = x[None] + 0.4

    foo()
    assert x[None] == approx(1.1)
    x[None] = 0.64
    assert x[None] == approx(0.64)
    x[None] = 0.66
    assert x[None] == approx(0.66)


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_fixed_matrix_rotation():
    qfxt = ti.types.quant.fixed(bits=16, max_value=1.2)

    x = ti.Matrix.field(2, 2, dtype=qfxt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x.get_scalar_field(0, 0), x.get_scalar_field(0, 1))
    ti.root.place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x.get_scalar_field(1, 0), x.get_scalar_field(1, 1))
    ti.root.place(bitpack)

    x[None] = [[1.0, 0.0], [0.0, 1.0]]

    @ti.kernel
    def rotate_18_degrees():
        angle = math.pi / 10
        x[None] = x[None] @ ti.Matrix(
            [[ti.cos(angle), ti.sin(angle)], [-ti.sin(angle),
                                              ti.cos(angle)]])

    for i in range(5):
        rotate_18_degrees()
    assert x[None][0, 0] == approx(0, abs=1e-4)
    assert x[None][0, 1] == approx(1, abs=1e-4)
    assert x[None][1, 0] == approx(-1, abs=1e-4)
    assert x[None][1, 1] == approx(0, abs=1e-4)


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_fixed_implicit_cast():
    qfxt = ti.types.quant.fixed(bits=13, scale=0.1)
    x = ti.field(dtype=qfxt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

    @ti.kernel
    def foo():
        x[None] = 10

    foo()
    assert x[None] == approx(10.0)


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_fixed_cache_read_only():
    qfxt = ti.types.quant.fixed(bits=15, scale=0.1)
    x = ti.field(dtype=qfxt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

    @ti.kernel
    def test(data: ti.f32):
        ti.cache_read_only(x)
        assert x[None] == data

    x[None] = 0.7
    test(0.7)
    x[None] = 1.2
    test(1.2)
