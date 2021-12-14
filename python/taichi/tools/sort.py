import taichi as ti


# Odd-even merge sort
# References:
# https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
# https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
def parallel_sort(x):
    N = x.shape[0]

    @ti.kernel
    def sort_stage(x: ti.template(), N: int, p: int, k: int, invocations: int):
        for inv in range(invocations):
            j = k % p + inv * 2 * k
            for i in range(0, min(k, N - j - k)):
                a = i + j
                b = i + j + k
                if int(a / (p * 2)) == int(b / (p * 2)):
                    val_a = x[a]
                    val_b = x[b]
                    if val_a > val_b:
                        x[a] = val_b
                        x[b] = val_a

    p = 1
    while p < N:
        k = p
        while k >= 1:
            invocations = int((N - k - k % p) / (2 * k)) + 1
            sort_stage(x, N, p, k, invocations)
            ti.sync()
            k = int(k / 2)
        p = int(p * 2)
