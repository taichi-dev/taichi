import taichi as ti
import numpy as np


@ti.test(arch=ti.cpu, debug=True, cfg_optimization=False)
def test_simple_array():
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

    # Test bit_struct SNode read and write in Python-scope by calling the wrapped, untranslated function body
    set_val.__wrapped__()
    verify_val.__wrapped__()


@ti.test(arch=ti.cpu, debug=True, cfg_optimization=False)
def test_custom_int_load_and_store():
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu14 = ti.type_factory_.get_custom_int_type(14, False)
    ci5 = ti.type_factory_.get_custom_int_type(5, True)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu14)
    z = ti.field(dtype=ci5)

    test_case_np = np.array(
        [[2**12 - 1, 2**14 - 1, -(2**3)], [2**11 - 1, 2**13 - 1, -(2**2)],
         [0, 0, 0], [123, 4567, 8], [10, 31, 11]],
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

    # Test bit_struct SNode read and write in Python-scope by calling the wrapped, untranslated function body
    for idx in range(len(test_case_np)):
        set_val.__wrapped__(idx)
        verify_val.__wrapped__(idx)


def test_bit_struct():
    def test_single_bit_struct(physical_type, compute_type, custom_bits,
                               test_case):
        ti.init(arch=ti.cpu,
                debug=True,
                print_ir=False,
                cfg_optimization=False)

        cit1 = ti.type_factory_.get_custom_int_type_with_compute_type(
            compute_type, custom_bits[0], True)
        cit2 = ti.type_factory_.get_custom_int_type_with_compute_type(
            compute_type, custom_bits[1], False)
        cit3 = ti.type_factory_.get_custom_int_type_with_compute_type(
            compute_type, custom_bits[2], True)

        a = ti.field(dtype=cit1)
        b = ti.field(dtype=cit2)
        c = ti.field(dtype=cit3)
        ti.root._bit_struct(num_bits=physical_type).place(a, b, c)

        @ti.kernel
        def set_val(test_val: ti.ext_arr()):
            a[None] = test_val[0]
            b[None] = test_val[1]
            c[None] = test_val[2]

        @ti.kernel
        def verify_val(test_val: ti.ext_arr()):
            assert a[None] == test_val[0]
            assert b[None] == test_val[1]
            assert c[None] == test_val[2]

        set_val(test_case)
        verify_val(test_case)

    test_single_bit_struct(8, 8, [3, 3, 2],
                           np.array([2**2 - 1, 2**3 - 1, -2**1]))
    test_single_bit_struct(16, 16, [4, 7, 5],
                           np.array([2**3 - 1, 2**7 - 1, -2**4]))
    test_single_bit_struct(32, 32, [17, 11, 4],
                           np.array([2**16 - 1, 2**10 - 1, -2**3]))
    test_single_bit_struct(64, 64, [32, 23, 9],
                           np.array([2**31 - 1, 2**23 - 1, -2**8]))
    test_single_bit_struct(32, 16, [7, 12, 13],
                           np.array([2**6 - 1, 2**12 - 1, -2**12]))
    test_single_bit_struct(64, 32, [18, 22, 24],
                           np.array([2**17 - 1, 2**22 - 1, -2**23]))

    test_single_bit_struct(16, 16, [5, 5, 6], np.array([15, 5, 20]))
    test_single_bit_struct(32, 32, [10, 10, 12], np.array([11, 19, 2020]))
