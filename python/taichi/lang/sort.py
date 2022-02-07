from taichi.lang.kernel_impl import kernel
from taichi.lang.runtime_ops import sync
from taichi.types.annotations import template


# Odd-even merge sort
# References:
# https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
# https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
@kernel
def sort_stage(keys: template(), use_values: int, values: template(), N: int,
               p: int, k: int, invocations: int):
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, min(k, N - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a]
                key_b = keys[b]
                if key_a > key_b:
                    keys[a] = key_b
                    keys[b] = key_a
                    if use_values != 0:
                        temp = values[a]
                        values[a] = values[b]
                        values[b] = temp


def parallel_sort(keys, values=None):
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
    print(num_stages)
