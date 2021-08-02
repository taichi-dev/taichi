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

    # May also want to suuport vector struct using compound types such as
    # st = ti.types.struct(a=ti.types.vector(3, ti.f32), b=ti.f32)
    # f = ti.field(dtype=st, shape=(n, ))

    x = ti.Struct.field({"a": ti.f32, "b": ti.f32}, shape=(n, ))
    y = ti.Struct.field({"a": ti.f32, "b": ti.f32})

    ti.root.dense(ti.i, n // 4).dense(ti.i, 4).place(y)

    @ti.kernel
    def init():
        for i in x:
            x[i].a = i
            y[i].a = i

    @ti.kernel
    def run_taichi_scope():
        for i in x:
            x[i].b = x[i].a

    def run_python_scope():
        for i in range(n):
            y[i].b = y[i].a * 2 + 1

    init()
    run_taichi_scope()
    for i in range(n):
        assert x[i].b == i
    run_python_scope()
    for i in range(n):
        assert y[i].b == i * 2 + 1
