import taichi as ti


@ti.all_archs
def benchmark_fill_scalar():
    a = ti.field(dtype=float, shape=())

    @ti.kernel
    def fill():
        a[None] = 1.0

    return ti.benchmark(fill, repeat=1000)
