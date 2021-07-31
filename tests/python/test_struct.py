import taichi as ti


@ti.all_archs
def test_linear():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.dense(ti.i, n).place(y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


def test_linear_repeated():
    for i in range(10):
        test_linear()


@ti.all_archs
def test_linear_nested():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(x)
    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@ti.all_archs
def test_linear_nested_aos():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(x, y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@ti.all_archs
def test_2d_nested():
    x = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.ij, n // 16).dense(ti.ij, (32, 16)).place(x)

    for i in range(n * 2):
        for j in range(n):
            x[i, j] = i + j * 10

    for i in range(n * 2):
        for j in range(n):
            assert x[i, j] == i + j * 10


@ti.all_archs
def test_custom_struct():
    n = 32

    st = ti.type_factory.make_struct(a=ti.i32, b=ti.f32)
    f = ti.field(dtype=st, shape=(n, ))

    @ti.kernel
    def init():
        for i in f:
            f[i].a = i
    @ti.kernel
    def run():
        for i in f:
            f[i].b = f[i].a
    init()
    run()
    for i in range(n):
        assert f[i].b == i
