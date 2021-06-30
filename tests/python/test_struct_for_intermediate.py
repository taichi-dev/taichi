import taichi as ti


@ti.test(require=ti.extension.sparse, demote_dense_struct_fors=False)
def test_nested():
    x = ti.field(ti.i32)

    p, q = 3, 7
    n, m = 2, 4

    ti.root.dense(ti.ij, (p, q)).dense(ti.ij, (n, m)).place(x)

    @ti.kernel
    def iterate():
        for i, j in x.parent():
            x[i, j] += 1

    iterate()
    for i in range(p):
        for j in range(q):
            assert x[i * n, j * m] == 1, (i, j)


@ti.test()
def test_nested_demote():
    x = ti.field(ti.i32)

    p, q = 3, 7
    n, m = 2, 4

    ti.root.dense(ti.ij, (p, q)).dense(ti.ij, (n, m)).place(x)

    @ti.kernel
    def iterate():
        for i, j in x.parent():
            x[i, j] += 1

    iterate()

    for i in range(p):
        for j in range(q):
            assert x[i * n, j * m] == 1
