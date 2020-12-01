import taichi as ti


@ti.test(require=ti.extension.quant, debug=True, cfg_optimization=False)
def test_custom_int_atomics():
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu2 = ti.type_factory_.get_custom_int_type(2, False)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu2)

    ti.root._bit_struct(num_bits=32).place(x, y)

    x[None] = 3
    y[None] = 0

    @ti.kernel
    def foo():
        for i in range(10):
            x[None] += 4

        for j in range(3):
            y[None] += 1

    foo()

    assert x[None] == 43
    assert y[None] == 3


@ti.test(require=ti.extension.quant, debug=True, cfg_optimization=False)
def test_custom_int_atomics_b64():
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu2 = ti.type_factory_.get_custom_int_type(2, False)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu2)

    ti.root._bit_struct(num_bits=64).place(x, y)

    x[None] = 3
    y[None] = 0

    @ti.kernel
    def foo():
        for i in range(10):
            x[None] += 4

        for j in range(3):
            y[None] += 1

    foo()

    assert x[None] == 43
    assert y[None] == 3
