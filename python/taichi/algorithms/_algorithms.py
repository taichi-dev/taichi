from taichi._kernels import (
    blit_from_field_to_field,
    scan_add_inclusive,
    sort_stage,
    uniform_add,
    warp_shfl_up_i32,
)
from taichi.lang.impl import current_cfg, field
from taichi.lang.kernel_impl import data_oriented
from taichi.lang.misc import cuda, vulkan
from taichi.lang.runtime_ops import sync
from taichi.lang.simt import subgroup
from taichi.types.primitive_types import i32


def parallel_sort(keys, values=None):
    """Odd-even merge sort

    References:
        https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
        https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
    """
    N = keys.shape[0]

    num_stages = 0
    p = 1
    while p < N:
        k = p
        while k >= 1:
            invocations = int((N - k - k % p) / (2 * k)) + 1
            if values is None:
                sort_stage(keys, 0, keys, N, p, k, invocations)
            else:
                sort_stage(keys, 1, values, N, p, k, invocations)
            num_stages += 1
            sync()
            k = int(k / 2)
        p = int(p * 2)


@data_oriented
class PrefixSumExecutor:
    """Parallel Prefix Sum (Scan) Helper

    Use this helper to perform an inclusive in-place's parallel prefix sum.

    References:
        https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
        https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
    """

    def __init__(self, length):
        self.sorting_length = length

        BLOCK_SZ = 64
        GRID_SZ = int((length + BLOCK_SZ - 1) / BLOCK_SZ)

        # Buffer position and length
        # This is a single buffer implementation for ease of aot usage
        ele_num = length
        self.ele_nums = [ele_num]
        start_pos = 0
        self.ele_nums_pos = [start_pos]

        while ele_num > 1:
            ele_num = int((ele_num + BLOCK_SZ - 1) / BLOCK_SZ)
            self.ele_nums.append(ele_num)
            start_pos += BLOCK_SZ * ele_num
            self.ele_nums_pos.append(start_pos)

        self.large_arr = field(i32, shape=start_pos)

    def run(self, input_arr):
        length = self.sorting_length
        ele_nums = self.ele_nums
        ele_nums_pos = self.ele_nums_pos

        if input_arr.dtype != i32:
            raise RuntimeError("Only ti.i32 type is supported for prefix sum.")

        if current_cfg().arch == cuda:
            inclusive_add = warp_shfl_up_i32
        elif current_cfg().arch == vulkan:
            inclusive_add = subgroup.inclusive_add
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for prefix sum.")

        blit_from_field_to_field(self.large_arr, input_arr, 0, length)

        # Kogge-Stone construction
        for i in range(len(ele_nums) - 1):
            if i == len(ele_nums) - 2:
                scan_add_inclusive(
                    self.large_arr,
                    ele_nums_pos[i],
                    ele_nums_pos[i + 1],
                    True,
                    inclusive_add,
                )
            else:
                scan_add_inclusive(
                    self.large_arr,
                    ele_nums_pos[i],
                    ele_nums_pos[i + 1],
                    False,
                    inclusive_add,
                )

        for i in range(len(ele_nums) - 3, -1, -1):
            uniform_add(self.large_arr, ele_nums_pos[i], ele_nums_pos[i + 1])

        blit_from_field_to_field(input_arr, self.large_arr, 0, length)


__all__ = ["parallel_sort", "PrefixSumExecutor"]
