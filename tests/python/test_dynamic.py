import taichi as ti


def ti_support_dynamic(test):
    return ti.all_archs(test)


def ti_support_non_top_dynamic(test):
    return ti.archs_excluding(ti.opengl)(test)


@ti_support_dynamic
def test_dynamic():
    x = ti.var(ti.f32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        pass

    for i in range(n):
        x[i] = i

    for i in range(n):
        assert x[i] == i


@ti_support_dynamic
def test_dynamic2():
    x = ti.var(ti.f32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            x[i] = i

    func()

    for i in range(n):
        assert x[i] == i


@ti_support_dynamic
def test_dynamic_matrix():
    x = ti.Matrix(2, 1, dt=ti.i32)
    n = 8192

    ti.root.dynamic(ti.i, n, chunk_size=128).place(x)

    @ti.kernel
    def func():
        ti.serialize()
        for i in range(n // 4):
            x[i * 4][1, 0] = i

    func()

    for i in range(n // 4):
        a = x[i * 4][1, 0]
        assert a == i
        if i + 1 < n // 4:
            b = x[i * 4 + 1][1, 0]
            assert b == 0


@ti_support_dynamic
def test_append():
    x = ti.var(ti.i32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            ti.append(x.parent(), [], i)

    func()

    elements = []
    for i in range(n):
        elements.append(x[i])
    elements.sort()
    for i in range(n):
        assert elements[i] == i


@ti_support_dynamic
def test_length():
    x = ti.var(ti.i32)
    y = ti.var(ti.f32, shape=())
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            ti.append(x.parent(), [], i)

    func()

    @ti.kernel
    def get_len():
        y[None] = ti.length(x.parent(), [])

    get_len()

    assert y[None] == n


@ti_support_dynamic
def test_append_ret_value():
    x = ti.var(ti.i32)
    y = ti.var(ti.i32)
    z = ti.var(ti.i32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)
    ti.root.dynamic(ti.i, n, 32).place(y)
    ti.root.dynamic(ti.i, n, 32).place(z)

    @ti.kernel
    def func():
        for i in range(n):
            u = ti.append(x.parent(), [], i)
            y[u] = i + 1
            z[u] = i + 3

    func()

    for i in range(n):
        assert x[i] + 1 == y[i]
        assert x[i] + 3 == z[i]


@ti_support_non_top_dynamic
def test_dense_dynamic():
    n = 128
    x = ti.var(ti.i32)
    l = ti.var(ti.i32, shape=n)

    ti.root.dense(ti.i, n).dynamic(ti.j, n, 8).place(x)

    @ti.kernel
    def func():
        ti.serialize()
        for i in range(n):
            for j in range(n):
                ti.append(x.parent(), j, i)

        for i in range(n):
            l[i] = ti.length(x.parent(), i)

    func()

    for i in range(n):
        assert l[i] == n


@ti_support_non_top_dynamic
def test_dense_dynamic_len():
    n = 128
    x = ti.var(ti.i32)
    l = ti.var(ti.i32, shape=n)

    ti.root.dense(ti.i, n).dynamic(ti.j, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            l[i] = ti.length(x.parent(), i)

    func()

    for i in range(n):
        assert l[i] == 0
