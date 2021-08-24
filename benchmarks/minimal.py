import taichi as ti


@ti.test()
def benchmark_fill_scalar():
    a = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def fill():
        a[None] = 1.0

    return ti.benchmark(fill, repeat=1000)
