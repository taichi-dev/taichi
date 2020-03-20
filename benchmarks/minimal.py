import taichi as ti


def benchmark_fill_scalar():
    a = ti.var(dt=ti.f32, shape=())

    @ti.kernel
    def fill():
        a[None] = 1.0

    return ti.benchmark(fill)
