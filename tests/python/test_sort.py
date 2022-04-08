import taichi as ti
from tests import test_utils


@test_utils.test(exclude=[ti.cc])
def test_sort():
    def test_sort_for_dtype(dtype, N):
        keys = ti.field(dtype, N)
        values = ti.field(dtype, N)

        @ti.kernel
        def fill():
            for i in keys:
                keys[i] = ti.random() * N
                values[i] = keys[i]

        fill()
        ti._kernels.parallel_sort(keys, values)

        keys_host = keys.to_numpy()
        values_host = values.to_numpy()

        for i in range(N):
            if i < N - 1:
                assert keys_host[i] <= keys_host[i + 1]
            assert keys_host[i] == values_host[i]

    test_sort_for_dtype(ti.i32, 1)
    test_sort_for_dtype(ti.i32, 256)
    test_sort_for_dtype(ti.i32, 100001)
    test_sort_for_dtype(ti.f32, 1)
    test_sort_for_dtype(ti.f32, 256)
    test_sort_for_dtype(ti.f32, 100001)
