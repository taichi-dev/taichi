import math

from pytest import approx

import taichi as ti


@ti.test(require=ti.extension.quant_basic)
def test_custom_float():
    cft = ti.quant.fixed(frac=32, range=2)
    x = ti.field(dtype=cft)

    ti.root.bit_struct(num_bits=32).place(x)

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


@ti.test(require=ti.extension.quant_basic)
def test_custom_matrix_rotation():
    cft = ti.quant.fixed(frac=16, range=1.2)

    x = ti.Matrix.field(2, 2, dtype=cft)

    ti.root.bit_struct(num_bits=32).place(x(0, 0), x(0, 1))
    ti.root.bit_struct(num_bits=32).place(x(1, 0), x(1, 1))

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


@ti.test(require=ti.extension.quant_basic)
def test_custom_float_implicit_cast():
    ci13 = ti.quant.int(bits=13)
    cft = ti.type_factory.custom_float(significand_type=ci13, scale=0.1)
    x = ti.field(dtype=cft)

    ti.root.bit_struct(num_bits=32).place(x)

    @ti.kernel
    def foo():
        x[None] = 10

    foo()
    assert x[None] == approx(10.0)


@ti.test(require=ti.extension.quant_basic)
def test_cache_read_only():
    ci15 = ti.quant.int(bits=15)
    cft = ti.type_factory.custom_float(significand_type=ci15, scale=0.1)
    x = ti.field(dtype=cft)

    ti.root.bit_struct(num_bits=32).place(x)

    @ti.kernel
    def test(data: ti.f32):
        ti.cache_read_only(x)
        assert x[None] == data

    x[None] = 0.7
    test(0.7)
    x[None] = 1.2
    test(1.2)
