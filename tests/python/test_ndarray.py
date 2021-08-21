import numpy as np
import pytest

import taichi as ti


@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_ndarray_2d():
    n = 4
    m = 7

    @ti.kernel
    def run(x: ti.any_arr(), y: ti.any_arr()):
        for i in range(n):
            for j in range(m):
                x[i, j] += i + j + y[i, j]

    a = ti.ndarray(ti.i32, shape=(n, m))
    for i in range(n):
        for j in range(m):
            a[i, j] = i * j
    b = np.ones((n, m), dtype=np.int32)
    run(a, b)
    for i in range(n):
        for j in range(m):
            assert a[i, j] == i * j + i + j + 1
    run(b, a)
    for i in range(n):
        for j in range(m):
            assert b[i, j] == i * j + (i + j + 1) * 2
