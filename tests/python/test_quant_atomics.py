from pytest import approx

import taichi as ti
from tests import test_utils


# TODO: remove excluding of ti.metal.
@test_utils.test(require=ti.extension.quant_basic,
                 exclude=[ti.metal],
                 debug=True)
def test_quant_int_atomics():
    qi13 = ti.types.quant.int(13, True)
    qi5 = ti.types.quant.int(5, True)
    qu2 = ti.types.quant.int(2, False)

    x = ti.field(dtype=qi13)
    y = ti.field(dtype=qi5)
    z = ti.field(dtype=qu2)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x, y, z)
    ti.root.place(bitpack)

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


@test_utils.test(require=[ti.extension.quant_basic, ti.extension.data64],
                 debug=True)
def test_quant_int_atomics_b64():
    qi13 = ti.types.quant.int(13, True)

    x = ti.field(dtype=qi13)

    ti.root.quant_array(ti.i, 4, max_num_bits=64).place(x)

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


@test_utils.test(require=ti.extension.quant_basic, debug=True)
def test_quant_fixed_atomics():
    qfxt13 = ti.types.quant.fixed(bits=13, signed=True, scale=0.1)
    qfxt19 = ti.types.quant.fixed(bits=19, signed=False, scale=0.1)

    x = ti.field(dtype=qfxt13)
    y = ti.field(dtype=qfxt19)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x, y)
    ti.root.place(bitpack)

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
