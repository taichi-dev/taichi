import taichi as ti


@ti.test(require=ti.extension.quant)
def test_custom_int_implicit_cast():
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    x = ti.field(dtype=ci13)

    ti.root._bit_struct(num_bits=32).place(x)

    @ti.kernel
    def foo():
        x[None] = 10.3

    foo()
    assert x[None] == 10
