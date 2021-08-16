import taichi as ti


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


@ti.torch_test
def test_ndarray_2d():
    n = 4
    m = 7
    a = ti.ndarray(ti.i32, shape=(n, m))
    _test_ndarray_2d(n, m, a)
