import taichi as ti
import numpy as np


@ti.test(arch=ti.cpu, debug=True, cfg_optimization=False)
def test_1D_bit_array():
    ci1 = ti.type_factory_.get_custom_int_type(1, False)

    x = ti.field(dtype=ci1)

    ti.root._bit_array(ti.i, 32, num_bits=32).place(x)

    ti.get_runtime().materialize()

    N = 32

    @ti.kernel
    def set_val():
        for i in range(N):
            x[i] = i % 2

    @ti.kernel
    def verify_val():
        for i in range(N):
            assert x[i] == i % 2

    set_val()
    verify_val()
