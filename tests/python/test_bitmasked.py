import taichi as ti


@ti.archs_excluding(ti.opengl)
def test_basic():
    x = ti.var(ti.i32)
    c = ti.var(ti.i32)
    s = ti.var(ti.i32)

    bm = ti.root.dense(ti.ij, (3, 6)).bitmasked().dense(ti.i, 5).bitmasked()
    bm.place(x)
    ti.root.place(c, s)

    @ti.kernel
    def run():
        x[5, 1] = 2
        x[9, 4] = 20
        x[0, 3] = 20

    @ti.kernel
    def sum():
        for i, j in x:
            ti.atomic_add(c[None], ti.is_active(bm, [i, j]))
            ti.atomic_add(s[None], x[i, j])

    run()
    sum()

    assert c[None] == 3
    assert s[None] == 42
