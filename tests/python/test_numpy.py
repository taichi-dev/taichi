import numpy as np

import taichi as ti


def with_data_type(dt):
    val = ti.field(ti.i32)

    n = 4

    ti.root.dense(ti.i, n).place(val)

    @ti.kernel
    def test_numpy(arr: ti.ext_arr()):
        for i in range(n):
            arr[i] = arr[i]**2

    a = np.array([4, 8, 1, 24], dtype=dt)

    for i in range(n):
        a[i] = i * 2

    test_numpy(a)

    for i in range(n):
        assert a[i] == i * i * 4


@ti.test()
def test_numpy_f32():
    with_data_type(np.float32)


@ti.test(require=ti.extension.data64)
def test_numpy_f64():
    with_data_type(np.float64)


@ti.test()
def test_numpy_i32():
    with_data_type(np.int32)


@ti.test(require=ti.extension.data64)
def test_numpy_i64():
    with_data_type(np.int64)


@ti.test()
def test_numpy_2d():
    val = ti.field(ti.i32)

    n = 4
    m = 7

    ti.root.dense(ti.i, n).dense(ti.j, m).place(val)

    @ti.kernel
    def test_numpy(arr: ti.ext_arr()):
        for i in range(n):
            for j in range(m):
                arr[i, j] += i + j

    a = np.empty(shape=(n, m), dtype=np.int32)

    for i in range(n):
        for j in range(m):
            a[i, j] = i * j

    test_numpy(a)

    for i in range(n):
        for j in range(m):
            assert a[i, j] == i * j + i + j


@ti.test()
def test_numpy_2d_transpose():
    val = ti.field(ti.i32)

    n = 8
    m = 8

    ti.root.dense(ti.ij, (n, m)).place(val)

    @ti.kernel
    def test_numpy(arr: ti.ext_arr()):
        for i in ti.grouped(val):
            val[i] = arr[i]

    a = np.empty(shape=(n, m), dtype=np.int32)

    for i in range(n):
        for j in range(m):
            a[i, j] = i * j + i * 4

    test_numpy(a.transpose())

    for i in range(n):
        for j in range(m):
            assert val[i, j] == i * j + j * 4


@ti.test()
def test_numpy_3d():
    val = ti.field(ti.i32)

    n = 4
    m = 7
    p = 11

    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

    @ti.kernel
    def test_numpy(arr: ti.ext_arr()):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    arr[i, j, k] += i + j + k * 2

    a = np.empty(shape=(n, m, p), dtype=np.int32)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a[i, j, k] = i * j * (k + 1)

    test_numpy(a)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert a[i, j, k] == i * j * (k + 1) + i + j + k * 2


@ti.test()
@ti.must_throw(IndexError)
def test_numpy_3d_error():
    val = ti.field(ti.i32)

    n = 4
    m = 7
    p = 11

    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

    @ti.kernel
    def test_numpy(arr: ti.ext_arr()):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    arr[i, j] += i + j + k * 2

    a = np.empty(shape=(n, m, p), dtype=np.int32)

    test_numpy(a)


@ti.test()
def test_numpy_multiple_external_arrays():

    n = 4

    @ti.kernel
    def test_numpy(a: ti.ext_arr(), b: ti.ext_arr()):
        for i in range(n):
            a[i] = a[i] * b[i]
            b[i] = a[i] + b[i]

    a = np.array([4, 8, 1, 24], dtype=np.int32)
    b = np.array([5, 6, 12, 3], dtype=np.int32)
    c = a * b
    d = c + b

    test_numpy(a, b)
    for i in range(n):
        assert a[i] == c[i]
        assert b[i] == d[i]


@ti.test()
@ti.must_throw(AssertionError)
def test_index_mismatch():
    val = ti.field(ti.i32, shape=(1, 2, 3))
    val[0, 0] = 1


@ti.test()
def test_numpy_zero():
    @ti.kernel
    def test_numpy(arr: ti.ext_arr()):
        pass

    test_numpy(np.empty(shape=(0), dtype=np.int32))
    test_numpy(np.empty(shape=(0, 5), dtype=np.int32))
    test_numpy(np.empty(shape=(5, 0), dtype=np.int32))


@ti.test()
def test_numpy_struct_for():
    @ti.kernel
    def func1(a: ti.any_arr()):
        for i, j in a:
            a[i, j] = i + j

    m = np.zeros((123, 456), dtype=np.int32)
    func1(m)
    for i in range(123):
        for j in range(456):
            assert m[i, j] == i + j

    @ti.kernel
    def func2(a: ti.any_arr()):
        for I in ti.grouped(a):
            a[I] = I.sum()

    n = np.zeros((98, 76, 54), dtype=np.int32)
    func2(n)
    for i, j, k in ti.ndrange(98, 76, 54):
        assert n[i, j, k] == i + j + k
