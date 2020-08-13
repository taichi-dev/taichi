import taichi as ti


@ti.require(ti.extension.sparse)
@ti.all_archs
def test_no_activate():
    x = ti.field(ti.f32)

    n = 1024

    d = ti.root.dynamic(ti.i, n, chunk_size=32)
    d.place(x)

    @ti.kernel
    def initialize():
        for i in range(n):
            x[i] = 1

    @ti.kernel
    def func():
        ti.no_activate(d)
        for i in range(n // 2):
            x[i * 2 + 1] += 1

    initialize()

    func()

    for i in range(n):
        assert x[i] == i % 2 + 1
