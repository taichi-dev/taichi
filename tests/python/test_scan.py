import pytest
import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
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

        # Performing an inclusive in-place's parallel prefix sum,
        # only one exectutor is needed for a specified sorting length.
        executor = ti.algorithms.PrefixSumExecutor(N)

        executor.run(arr)

        cur_sum = 0
        for i in range(N):
            cur_sum += arr_aux[i]
            assert arr[i] == cur_sum

    test_scan_for_dtype(ti.i32, 512)
    test_scan_for_dtype(ti.i32, 1024)
    test_scan_for_dtype(ti.i32, 4096)


@pytest.mark.parametrize("dtype", [ti.i32])
@pytest.mark.parametrize("N", [512, 1024, 4096])
@pytest.mark.parametrize("offset", [0, -1, 1, 256, -256, -23333, 23333])
@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scan_with_offset(dtype, N, offset):
    arr = ti.field(dtype, N, offset=offset)
    arr_aux = ti.field(dtype, N, offset=offset)

    @ti.kernel
    def fill():
        for i in arr:
            arr[i] = ti.random() * N
            arr_aux[i] = arr[i]

    fill()

    # Performing an inclusive in-place's parallel prefix sum,
    # only one exectutor is needed for a specified sorting length.
    executor = ti.algorithms.PrefixSumExecutor(N)

    executor.run(arr)

    cur_sum = 0
    for i in range(N):
        cur_sum += arr_aux[i + offset]
        assert arr[i + offset] == cur_sum
