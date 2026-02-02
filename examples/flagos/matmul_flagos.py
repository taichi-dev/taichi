"""
Taichi FlagOS Backend Example: Matrix Multiplication

This example demonstrates matrix multiplication using the FlagOS backend,
which is a common workload in AI computing.
"""

import taichi as ti
import numpy as np
import time

# Initialize with FlagOS backend
ti.init(arch=ti.flagos if hasattr(ti, "flagos") else ti.cpu, flagos_chip="mlu370", kernel_profiler=True)

# Matrix dimensions
N = 1024

# Taichi fields
A = ti.field(dtype=ti.f32, shape=(N, N))
B = ti.field(dtype=ti.f32, shape=(N, N))
C = ti.field(dtype=ti.f32, shape=(N, N))


@ti.kernel
def matmul_tiled():
    """Tiled matrix multiplication kernel"""
    for i, j in ti.ndrange(N, N):
        sum = 0.0
        for k in range(N):
            sum += A[i, k] * B[k, j]
        C[i, j] = sum


@ti.kernel
def matmul_blocked(BLOCK_SIZE: ti.template()):
    """Blocked matrix multiplication for better memory locality"""
    for i, j in ti.ndrange(N // BLOCK_SIZE, N // BLOCK_SIZE):
        for ii, jj in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
            row = i * BLOCK_SIZE + ii
            col = j * BLOCK_SIZE + jj
            sum = 0.0
            for k in range(N):
                sum += A[row, k] * B[k, col]
            C[row, col] = sum


def benchmark_matmul():
    """Benchmark matrix multiplication"""
    # Initialize matrices with random values
    A_np = np.random.randn(N, N).astype(np.float32)
    B_np = np.random.randn(N, N).astype(np.float32)

    A.from_numpy(A_np)
    B.from_numpy(B_np)

    # Warmup
    print("Warming up...")
    matmul_tiled()
    ti.sync()

    # Benchmark
    print(f"\nBenchmarking {N}x{N} matrix multiplication...")

    # Simple tiled version
    start = time.time()
    matmul_tiled()
    ti.sync()
    elapsed = time.time() - start

    print(f"\nSimple tiled version:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  GFLOPS: {2.0 * N**3 / elapsed / 1e9:.2f}")

    # Verify correctness
    C_np = C.to_numpy()
    C_ref = A_np @ B_np
    error = np.max(np.abs(C_np - C_ref))
    print(f"  Max error: {error:.6f}")

    # Print kernel profile
    print("\nKernel profiling:")
    ti.profiler.print_kernel_profiler_info()


def main():
    print("Taichi FlagOS Backend - Matrix Multiplication Benchmark")
    print("=" * 60)
    print(f"Matrix size: {N}x{N}")
    print(f"Target chip: {ti.cfg.flagos_chip if hasattr(ti.cfg, 'flagos_chip') else 'N/A'}")
    print()

    benchmark_matmul()


if __name__ == "__main__":
    main()
