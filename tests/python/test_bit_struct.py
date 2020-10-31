import taichi as ti
import numpy as np


def test_simple_array():
    ti.init(arch=ti.cpu, debug=True, print_ir=True, cfg_optimization=False)
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu19 = ti.type_factory_.get_custom_int_type(19, False)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu19)

    N = 12

    ti.root.dense(ti.i, N)._bit_struct(num_bits=32).place(x, y)

    ti.get_runtime().materialize()

    @ti.kernel
    def set_val():
        for i in range(N):
            x[i] = -2**i
            y[i] = 2**i - 1

    @ti.kernel
    def verify_val():
        for i in range(N):
            assert x[i] == -2**i
            assert y[i] == 2**i - 1

    set_val()
    verify_val()


def test_custom_int_load_and_store():
    ti.init(arch=ti.cpu, debug=True, print_ir=True, cfg_optimization=False)
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu14 = ti.type_factory_.get_custom_int_type(14, False)
    ci5 = ti.type_factory_.get_custom_int_type(5, True)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu14)
    z = ti.field(dtype=ci5)

    test_case_np = np.array([[2**12 - 1, 2**14 - 1, -(2**3)],
                             [2**11 - 1, 2**13 - 1, -(2**2)], [0, 0, 0],
                             [123, 4567, 8],
                             [10, 31, 11]],
                            dtype=np.int32)

    ti.root._bit_struct(num_bits=32).place(x, y, z)
    test_case = ti.Vector.field(3, dtype=ti.i32, shape=len(test_case_np))
    test_case.from_numpy(test_case_np)

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
