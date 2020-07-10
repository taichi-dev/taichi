import taichi as ti
import pytest

# originally by @KLozes


@ti.all_archs
@pytest.mark.benchmark(min_rounds=50)
def test_flat_struct(benchmark):
    N = 4096
    a = ti.var(dt=ti.f32, shape=(N, N))

    @ti.kernel
    def fill():
        for i, j in a:
            a[i, j] = 2.0

    return benchmark(fill)


@ti.all_archs
@pytest.mark.benchmark(min_rounds=70)
def test_flat_range(benchmark):
    N = 4096
    a = ti.var(dt=ti.f32, shape=(N, N))

    @ti.kernel
    def fill():
        for i, j in ti.ndrange(N, N):
            a[i, j] = 2.0

    return benchmark(fill)


@ti.all_archs
@pytest.mark.benchmark(min_rounds=70)
def test_nested_struct(benchmark):
    a = ti.var(dt=ti.f32)
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in a:
            a[i, j] = 2.0

    return benchmark(fill)


@ti.all_archs
@pytest.mark.benchmark(min_rounds=100)
def test_nested_struct_listgen_8x8(benchmark):
    a = ti.var(dt=ti.f32)
    ti.cfg.demote_dense_struct_fors = False
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in a:
            a[i, j] = 2.0

    return benchmark(fill)


@ti.all_archs
@pytest.mark.benchmark(min_rounds=70)
def test_nested_struct_listgen_16x16(benchmark):
    a = ti.var(dt=ti.f32)
    ti.cfg.demote_dense_struct_fors = False
    N = 256

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [16, 16]).place(a)

    @ti.kernel
    def fill():
        for i, j in a:
            a[i, j] = 2.0

    return benchmark(fill)


@ti.all_archs
@pytest.mark.benchmark(min_rounds=80)
def test_nested_range_blocked(benchmark):
    a = ti.var(dt=ti.f32)
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for X in range(N * N):
            for Y in range(64):
                a[X // N * 8 + Y // 8, X % N * 8 + Y % 8] = 2.0

    return benchmark(fill)


@ti.all_archs
@pytest.mark.benchmark(min_rounds=100)
def test_nested_range(benchmark):
    a = ti.var(dt=ti.f32)
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for j in range(N * 8):
            for i in range(N * 8):
                a[i, j] = 2.0

    return benchmark(fill)


@ti.all_archs
@pytest.mark.benchmark(min_rounds=80)
def test_root_listgen(benchmark):
    a = ti.var(dt=ti.f32)
    ti.cfg.demote_dense_struct_fors = False
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in a.parent():
            a[i, j] = 2.0

    return benchmark(fill)


'''
# ti.cfg.arch = ti.cuda
# ti.cfg.print_kernel_llvm_ir_optimized = True
# ti.cfg.print_kernel_llvm_ir = True
ti.cfg.kernel_profiler = True
# ti.cfg.verbose_kernel_launches = True
print(benchmark_nested_struct_listgen_8x8())
# print(benchmark_root_listgen())
ti.kernel_profiler_print()
'''
