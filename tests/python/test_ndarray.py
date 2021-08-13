import taichi as ti

import numpy as np
if ti.has_pytorch():
    import torch


def _test_ndarray_2d(n, m, a):
    @ti.kernel
    def run(arr: ti.ext_arr()):
        for i in range(n):
            for j in range(m):
                arr[i, j] += i + j

    for i in range(n):
        for j in range(m):
            a[i, j] = i * j

    run(a)

    for i in range(n):
        for j in range(m):
            assert a[i, j] == i * j + i + j


@ti.test()
def test_ndarray_numpy_2d():
    n = 4
    m = 7
    a = ti.Ndarray(np.empty(shape=(n, m), dtype=np.int32))
    _test_ndarray_2d(n, m, a)


@ti.torch_test
def test_ndarray_torch_2d():
    n = 4
    m = 7
    a = ti.Ndarray(torch.empty((n, m), dtype=torch.int32))
    _test_ndarray_2d(n, m, a)


@ti.torch_test
def test_ndarray_default_2d():
    n = 4
    m = 7
    a = ti.ndarray(ti.i32, shape=(n, m))
    _test_ndarray_2d(n, m, a)
