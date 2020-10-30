import taichi as ti


def test_custom_int_load_and_store():
    ti.init(arch=ti.cpu, debug=True, advanced_optimization=False)
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu14 = ti.type_factory_.get_custom_int_type(14, False)
    ci5 = ti.type_factory_.get_custom_int_type(5, True)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu14)
    z = ti.field(dtype=ci5)

    ti.root._bit_struct(num_bits=32).place(x, y, z)

    ti.get_runtime().materialize()

    @ti.kernel
    def foo():
        x[None] = 2**12-1
        assert x[None] == 2**12-1
        y[None] = 2**14-1
        assert y[None] == 2**14-1
        z[None] = -(2**3)
        assert z[None] == -(2**3)

    foo()

