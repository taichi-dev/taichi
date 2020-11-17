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


@ti.test(arch=ti.cpu, debug=True, cfg_optimization=False)
def test_bit_struct_with_physical_type():
    ci8_5 = ti.type_factory_._get_custom_int_type(8, 5, True)
    ci8_3 = ti.type_factory_._get_custom_int_type(8, 3, False)
    ci16_4 = ti.type_factory_._get_custom_int_type(16, 4, True)
    cu16_12 = ti.type_factory_._get_custom_int_type(16, 12, False)

    ci32_17 = ti.type_factory_._get_custom_int_type(32, 17, True)
    ci32_11 = ti.type_factory_._get_custom_int_type(32, 11, True)
    cu32_4 = ti.type_factory_._get_custom_int_type(32, 4, False)

    ci64_33 = ti.type_factory_._get_custom_int_type(64, 32, True)
    ci64_20 = ti.type_factory_._get_custom_int_type(64, 21, False)
    ci64_7 = ti.type_factory_._get_custom_int_type(64, 7, False)

    a = ti.field(dtype=ci8_5)
    b = ti.field(dtype=ci8_3)

    c = ti.field(dtype=ci16_4)
    d = ti.field(dtype=cu16_12)

    e = ti.field(dtype=ci32_17)
    f = ti.field(dtype=ci32_11)
    g = ti.field(dtype=cu32_4)

    h = ti.field(dtype=ci64_33)
    i = ti.field(dtype=ci64_20)
    j = ti.field(dtype=ci64_7)

    k = ti.field(dtype=ci16_4)
    l = ti.field(dtype=cu16_12)

    m = ti.field(dtype=ci32_17)
    n = ti.field(dtype=ci32_11)
    o = ti.field(dtype=cu32_4)

    ti.root._bit_struct(num_bits=8).place(a, b)
    ti.root._bit_struct(num_bits=16).place(c, d)
    ti.root._bit_struct(num_bits=32).place(e, f, g)
    ti.root._bit_struct(num_bits=64).place(h, i, j)

    ti.root._bit_struct(num_bits=32).place(k, l)
    ti.root._bit_struct(num_bits=64).place(m, n, o)

    test_case_np = np.array(
        [[2**4 - 1, 2**3 - 1, 2**3 - 1, 2**12 - 1, 2**16-1, 2**10-1, 2**4-1, 2**31-1, 2**21-1, 2**7-1, 2**3 - 1, 2**12 - 1, 2**16-1, 2**10-1, 2**4-1],
         [-2**3, 2**2 - 1, -2**2, 2**11 - 1, -2**15, -2**9, 2**2-1, -2**30, 2**20-1, 2**6-1, -2**2, 2**11 - 1, -2**15, -2**9, 2**2-1,],
         [3, 4, 5, 16, 21, 34, 1, 2020, 456, 123, 5, 16, 21, 34, 1]],
        dtype=np.int32)
    test_case = ti.Vector.field(15, dtype=ti.i32, shape=len(test_case_np))
    test_case.from_numpy(test_case_np)

    @ti.kernel
    def set_val(idx: ti.i32):
        a[None] = test_case[idx][0]
        b[None] = test_case[idx][1]
        c[None] = test_case[idx][2]
        d[None] = test_case[idx][3]
        e[None] = test_case[idx][4]
        f[None] = test_case[idx][5]
        g[None] = test_case[idx][6]
        h[None] = test_case[idx][7]
        i[None] = test_case[idx][8]
        j[None] = test_case[idx][9]
        k[None] = test_case[idx][10]
        l[None] = test_case[idx][11]
        m[None] = test_case[idx][12]
        n[None] = test_case[idx][13]
        o[None] = test_case[idx][14]

    @ti.kernel
    def verify_val(idx: ti.i32):
        assert a[None] == test_case[idx][0]
        assert b[None] == test_case[idx][1]
        assert c[None] == test_case[idx][2]
        assert d[None] == test_case[idx][3]
        assert e[None] == test_case[idx][4]
        assert f[None] == test_case[idx][5]
        assert g[None] == test_case[idx][6]
        assert h[None] == test_case[idx][7]
        assert i[None] == test_case[idx][8]
        assert j[None] == test_case[idx][9]
        assert k[None] == test_case[idx][10]
        assert l[None] == test_case[idx][11]
        assert m[None] == test_case[idx][12]
        assert n[None] == test_case[idx][13]
        assert o[None] == test_case[idx][14]

    for idx in range(len(test_case_np)):
        set_val(idx)
        verify_val(idx)
