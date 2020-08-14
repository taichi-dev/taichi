import taichi as ti


@ti.all_archs
def test_vector_index():
    val = ti.field(ti.i32)

    n = 4
    m = 7
    p = 11

    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

    @ti.kernel
    def test():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = ti.Vector([i, j, k])
                    val[I] = i + j * 2 + k * 3

    test()

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert val[i, j, k] == i + j * 2 + k * 3


@ti.all_archs
def test_grouped():
    val = ti.field(ti.i32)

    n = 4
    m = 8
    p = 16

    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

    @ti.kernel
    def test():
        for I in ti.grouped(val):
            val[I] = I[0] + I[1] * 2 + I[2] * 3

    test()

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert val[i, j, k] == i + j * 2 + k * 3


@ti.all_archs
def test_grouped_ndrange():
    val = ti.field(ti.i32)

    n = 4
    m = 8

    ti.root.dense(ti.ij, (n, m)).place(val)

    x0 = 2
    y0 = 3
    x1 = 1
    y1 = 6

    @ti.kernel
    def test():
        for I in ti.grouped(ti.ndrange((x0, y0), (x1, y1))):
            val[I] = I[0] + I[1] * 2

    test()

    for i in range(n):
        for j in range(m):
            assert val[i, j] == (i +
                                 j * 2 if x0 <= i < y0 and x1 <= j < y1 else 0)


@ti.all_archs
def test_static_grouped_ndrange():
    val = ti.field(ti.i32)

    n = 4
    m = 8

    ti.root.dense(ti.ij, (n, m)).place(val)

    x0 = 2
    y0 = 3
    x1 = 1
    y1 = 6

    @ti.kernel
    def test():
        for I in ti.static(ti.grouped(ti.ndrange((x0, y0), (x1, y1)))):
            val[I] = I[0] + I[1] * 2

    test()

    for i in range(n):
        for j in range(m):
            assert val[i, j] == (i +
                                 j * 2 if x0 <= i < y0 and x1 <= j < y1 else 0)


@ti.all_archs
def test_grouped_ndrange_starred():
    val = ti.field(ti.i32)

    n = 4
    m = 8
    p = 16
    dim = 3

    ti.root.dense(ti.ijk, (n, m, p)).place(val)

    @ti.kernel
    def test():
        for I in ti.grouped(ti.ndrange(*(((0, n), ) * dim))):
            val[I] = I[0] + I[1] * 2 + I[2] * 3

    test()

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert val[i, j,
                           k] == (i + j * 2 + k * 3 if j < n and k < n else 0)


@ti.all_archs
def test_grouped_ndrange_0d():
    val = ti.field(ti.i32, shape=())

    @ti.kernel
    def test():
        for I in ti.grouped(ti.ndrange()):
            val[I] = 42

    test()

    assert val[None] == 42


@ti.all_archs
def test_static_grouped_ndrange_0d():
    val = ti.field(ti.i32, shape=())

    @ti.kernel
    def test():
        for I in ti.static(ti.grouped(ti.ndrange())):
            val[I] = 42

    test()

    assert val[None] == 42


@ti.all_archs
def test_static_grouped_func():

    K = 3
    dim = 2

    v = ti.Vector.field(K, dtype=ti.i32, shape=((K, ) * dim))

    def stencil_range():
        return ti.ndrange(*((K, ) * (dim + 1)))

    @ti.kernel
    def p2g():
        for I in ti.static(ti.grouped(stencil_range())):
            v[I[0], I[1]][I[2]] = I[0] + I[1] * 3 + I[2] * 10

    p2g()

    for i in range(K):
        for j in range(K):
            for k in range(K):
                assert v[i, j][k] == i + j * 3 + k * 10
