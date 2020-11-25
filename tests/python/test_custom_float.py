import taichi as ti
from pytest import approx


@ti.test(arch=ti.cpu, debug=True, cfg_optimization=False)
def test_custom_float():
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cft = ti.type_factory_.get_custom_float_type(ci13, ti.f32.get_ptr(), 0.1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    ti.get_runtime().print_snode_tree()
    ti.get_runtime().materialize()

    @ti.kernel
    def foo():
        x[None] = 0.7
        print(x[None])
        x[None] = x[None] + 0.4

    foo()
    assert x[None] == approx(1.1)
    x[None] = 0.64
    assert x[None] == approx(0.6)
    x[None] = 0.66
    assert x[None] == approx(0.7)
