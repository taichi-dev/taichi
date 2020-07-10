import taichi as ti


@ti.all_archs
def test_fill_scalar(benchmark):
    a = ti.var(dt=ti.f32, shape=())

    @ti.kernel
    def fill():
        a[None] = 1.0

    return benchmark(fill)
