import taichi as ti
import pytest


@ti.archs_support_sparse
def test_nested_struct(benchmark):
    a = ti.var(dt=ti.f32)
    N = 512

    ti.root.pointer(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in ti.ndrange(N * 8, N * 8):
            a[i, j] = 2.0

    fill()

    return benchmark(fill)


@ti.archs_support_sparse
@pytest.mark.benchmark(min_rounds=30)
def test_nested_struct_fill_and_clear(benchmark):
    a = ti.var(dt=ti.f32)
    N = 512

    ti.root.pointer(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in ti.ndrange(N * 8, N * 8):
            a[i, j] = 2.0

    @ti.kernel
    def clear():
        for i, j in a.parent():
            ti.deactivate(a.parent().parent(), [i, j])

    def task():
        fill()
        clear()

    return benchmark(task)


'''
ti.init(arch=ti.cuda, kernel_profiler=True)
benchmark_nested_struct_fill_and_clear()
ti.kernel_profiler_print()
'''
