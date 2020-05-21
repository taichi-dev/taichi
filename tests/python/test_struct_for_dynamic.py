import taichi as ti


@ti.archs_support_sparse
def test_dynamic():
    x = ti.var(ti.i32)
    y = ti.var(ti.i32, shape=())

    n = 128

    @ti.layout
    def place():
        ti.root.dynamic(ti.i, n).place(x)

    @ti.kernel
    def count():
        for i in x:
            y[None] += 1

    x[n // 3] = 1

    count()

    assert y[None] == n // 3 + 1


@ti.archs_support_sparse
def test_dense_dynamic():
    n = 128

    x = ti.var(ti.i32)

    @ti.layout
    def place():
        ti.root.dense(ti.i, n).dynamic(ti.j, n, 128).place(x)

    @ti.kernel
    def append():
        for i in range(n):
            for j in range(i):
                ti.append(x.parent(), i, j * 2)

    append()

    for i in range(n):
        for j in range(i):
            assert x[i, j] == j * 2
