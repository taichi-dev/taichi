import taichi as ti


@ti.all_archs
def test_simple():
    # Note: access simplification does not work in this case. Maybe worth fixing.
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n).place(x, y)

    @ti.kernel
    def run():
        for i in range(n - 1):
            x[i] = 1
            y[i + 1] = 2

    run()

    for i in range(n - 1):
        assert x[i] == 1
        assert y[i + 1] == 2
