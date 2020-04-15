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


@archs_support_bitmasked
def test_huge_bitmasked():
    # Mainly for testing Metal listgen's grid-stride loop implementation.
    x = ti.var(ti.f32)
    s = ti.var(ti.i32)

    n = 1024

    ti.root.bitmasked(ti.i, n).bitmasked(ti.i, 2 * n).place(x)
    ti.root.place(s)

    @ti.kernel
    def func():
        for i in range(n * n * 2):
            if i % 32 == 0:
                x[i] = 1.0

    @ti.kernel
    def count():
        for i in x:
            s[None] += 1

    func()
    count()
    assert s[None] == (n * n * 2) // 32


@archs_support_bitmasked
def test_bitmasked_listgen_bounded():
    # Mainly for testing Metal's listgen is bounded by the actual number of
    # elements possible for that SNode. Note that 1) SNode's size is padded
    # to POT, and 2) Metal ListManager's data size is not padded, we need to
    # make sure listgen doesn't go beyond ListManager's capacity.
    x = ti.var(ti.i32)
    c = ti.var(ti.i32)

    # A prime that is bit higher than 65536, which is Metal's maximum number of
    # threads for listgen.
    n = 80173

    ti.root.dense(ti.i, n).bitmasked(ti.i, 1).place(x)
    ti.root.place(c)

    @ti.kernel
    def func():
        for i in range(n):
            x[i] = 1

    @ti.kernel
    def count():
        for i in x:
            c[None] += 1

    func()
    count()
    assert c[None] == n


@archs_support_bitmasked
def test_deactivate():
    # https://github.com/taichi-dev/taichi/issues/778
    a = ti.var(ti.i32)
    a_a = ti.root.bitmasked(ti.i, 4)
    a_b = a_a.dense(ti.i, 4)
    a_b.place(a)
    c = ti.var(ti.i32)
    ti.root.place(c)

    @ti.kernel
    def run():
        a[0] = 123

    @ti.kernel
    def is_active():
        c[None] = ti.is_active(a_a, [0])

    @ti.kernel
    def deactivate():
        ti.deactivate(a_a, [0])

    run()
    is_active()
    assert c[None] == 1

    deactivate()
    is_active()
    assert c[None] == 0
