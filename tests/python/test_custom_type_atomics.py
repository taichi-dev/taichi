from pytest import approx

import taichi as ti


@ti.test(require=ti.extension.quant_basic, debug=True)
def test_custom_int_atomics():
    ci13 = ti.quant.int(13, True)
    ci5 = ti.quant.int(5, True)
    cu2 = ti.quant.int(2, False)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=ci5)
    z = ti.field(dtype=cu2)

    ti.root.bit_struct(num_bits=32).place(x, y, z)

    x[None] = 3
    y[None] = 2
    z[None] = 0

    @ti.kernel
    def foo():
        for i in range(10):
            x[None] += 4

        for j in range(5):
            y[None] -= 1

        for k in range(3):
            z[None] += 1

    foo()

    assert x[None] == 43
    assert y[None] == -3
    assert z[None] == 3


@ti.test(require=[ti.extension.quant_basic, ti.extension.data64], debug=True)
def test_custom_int_atomics_b64():
    ci13 = ti.quant.int(13, True)

    x = ti.field(dtype=ci13)

    ti.root.bit_array(ti.i, 4, num_bits=64).place(x)

    x[0] = 100
    x[1] = 200
    x[2] = 300

    @ti.kernel
    def foo():
        for i in range(9):
            x[i % 3] += i

    foo()

    assert x[0] == 109
    assert x[1] == 212
    assert x[2] == 315


@ti.test(require=ti.extension.quant_basic, debug=True)
def test_custom_float_atomics():
    ci13 = ti.quant.int(13, True)
    ci19 = ti.quant.int(19, False)
    cft13 = ti.type_factory.custom_float(significand_type=ci13, scale=0.1)
    cft19 = ti.type_factory.custom_float(significand_type=ci19, scale=0.1)

    x = ti.field(dtype=cft13)
    y = ti.field(dtype=cft19)

    ti.root.bit_struct(num_bits=32).place(x, y)

    @ti.kernel
    def foo():
        x[None] = 0.7
        y[None] = 123.4
        for _ in range(10):
            x[None] -= 0.4
            y[None] += 100.1

    foo()
    assert x[None] == approx(-3.3)
    assert y[None] == approx(1124.4)
