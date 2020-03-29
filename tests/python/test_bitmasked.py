import taichi as ti


def archs_support_bitmasked(func):
    return ti.archs_excluding(ti.opengl)(func)


@archs_support_bitmasked
def test_basic():
    x = ti.var(ti.i32)
    c = ti.var(ti.i32)
    s = ti.var(ti.i32)

    bm = ti.root.bitmasked(ti.ij, (3, 6)).bitmasked(ti.i, 5)
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
            c[None] += ti.is_active(bm, [i, j])
            s[None] += x[i, j]

    run()
    sum()

    assert c[None] == 3
    assert s[None] == 42


@archs_support_bitmasked
def test_bitmasked_then_dense():
    x = ti.var(ti.f32)
    s = ti.var(ti.i32)

    n = 128

    @ti.layout
    def place():
        ti.root.bitmasked(ti.i, n).dense(ti.i, n).place(x)
        ti.root.place(s)

    @ti.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[127] = 1
    x[256] = 1
    x[257] = 1

    func()
    assert s[None] == 256


@archs_support_bitmasked
def test_bitmasked_bitmasked():
    x = ti.var(ti.f32)
    s = ti.var(ti.i32)

    n = 128

    ti.root.bitmasked(ti.i, n).bitmasked(ti.i, n).place(x)
    ti.root.place(s)

    @ti.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[127] = 1
    x[256] = 1
    x[257] = 1

    func()
    assert s[None] == 4
