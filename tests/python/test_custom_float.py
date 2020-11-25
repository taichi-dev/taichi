import taichi as ti
import numpy as np


# @ti.test(arch=ti.cpu, debug=True, cfg_optimization=False)
def test_custom_float_load():
    ci13 = ti.type_factory_.get_custom_int_type(13, True)

    cft = ti.type_factory_.get_custom_float_type(ci13, ti.f32.get_ptr(), 0.1)
    x = ti.field(dtype=cft)
    # x = ti.field(dtype=ci13)

    ti.root._bit_struct(num_bits=32).place(x)

    ti.get_runtime().print_snode_tree()
    ti.get_runtime().materialize()

    @ti.kernel
    def foo():
        x[None] = 0.7
        print(x[None])

    foo()
    print(x[None])
    return


ti.init(arch=ti.cpu,
        debug=True,
        cfg_optimization=False,
        print_ir=True,
        print_accessor_ir=True)
test_custom_float_load()
