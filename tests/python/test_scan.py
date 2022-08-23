import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_scan():
    def test_scan_for_dtype(dtype, N):
        arr = ti.field(dtype, N)
        arr_aux = ti.field(dtype, N)

        @ti.kernel
        def fill():
            for i in arr:
                arr[i] = ti.random() * N
                arr_aux[i] = arr[i]

        fill()
        ti._kernels.prefix_sum_inclusive_inplace(arr, N)

        cur_sum = 0
        for i in range(N):
            cur_sum += arr_aux[i]
            assert arr[i] == cur_sum

    test_scan_for_dtype(ti.i32, 512)
    test_scan_for_dtype(ti.i32, 1024)
    test_scan_for_dtype(ti.i32, 4096)
