import taichi as ti
import numpy as np


@ti.test(arch=ti.cpu, debug=True, cfg_optimization=False)
def test_custom_float_load():
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cft = ti.type_factory_.get_custom_float_type(ci13, ti.f32.get_ptr(), 0.1)

    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    ti.get_runtime().materialize()

    return

    @ti.kernel
    def set_val(idx: ti.i32):
        x[None] = test_case[idx][0]
        y[None] = test_case[idx][1]
        z[None] = test_case[idx][2]

    @ti.kernel
    def verify_val(idx: ti.i32):
        assert x[None] == test_case[idx][0]
        assert y[None] == test_case[idx][1]
        assert z[None] == test_case[idx][2]

    for idx in range(len(test_case_np)):
        set_val(idx)
        verify_val(idx)
    '''
    # Test bit_struct SNode read and write in Python-scope by calling the wrapped, untranslated function body
    for idx in range(len(test_case_np)):
        set_val.__wrapped__(idx)
        verify_val.__wrapped__(idx)
    '''
