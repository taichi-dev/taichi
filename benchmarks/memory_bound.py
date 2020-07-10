import taichi as ti
import pytest

N = 1024**3 // 4  # 1 GB per buffer


# 4 B/it
@ti.archs_excluding(ti.opengl)
@pytest.mark.benchmark(min_rounds=10)
def test_memset(benchmark):
    a = ti.var(dt=ti.f32, shape=N)

    @ti.kernel
    def memset():
        for i in a:
            a[i] = 1.0

    return benchmark(memset)


# 8 B/it
@ti.archs_excluding(ti.opengl)
@pytest.mark.benchmark(min_rounds=10)
def test_sscal(benchmark):
    a = ti.var(dt=ti.f32, shape=N)

    @ti.kernel
    def task():
        for i in a:
            a[i] = 0.5 * a[i]

    return benchmark(task)


# 8 B/it
@ti.archs_excluding(ti.opengl)
@pytest.mark.benchmark(min_rounds=10)
def test_memcpy(benchmark):
    a = ti.var(dt=ti.f32, shape=N)
    b = ti.var(dt=ti.f32, shape=N)

    @ti.kernel
    def memcpy():
        for i in a:
            a[i] = b[i]

    return benchmark(memcpy)


# 12 B/it
@ti.archs_excluding(ti.opengl)
@pytest.mark.benchmark(min_rounds=10)
def test_saxpy(benchmark):
    x = ti.var(dt=ti.f32, shape=N)
    y = ti.var(dt=ti.f32, shape=N)
    z = ti.var(dt=ti.f32, shape=N)

    @ti.kernel
    def task():
        for i in x:
            a = 123
            z[i] = a * x[i] + y[i]

    return benchmark(task)
