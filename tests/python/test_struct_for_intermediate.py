import taichi as ti


@ti.all_archs
def test_nested():
    ti.cfg.demote_dense_struct_fors = False
    x = ti.var(ti.i32)

    p, q = 3, 7
    n, m = 2, 4

    @ti.layout
    def place():
        ti.root.dense(ti.ij, (p, q)).dense(ti.ij, (n, m)).place(x)

    @ti.kernel
    def iterate():
        for i, j in x.parent():
            x[i, j] += 1

    iterate()

    for i in range(p):
        for j in range(q):
            assert x[i * n, j * m] == 1


@ti.all_archs
def test_nested_demote():
    ti.cfg.demote_dense_struct_fors = True
    ti.cfg.print_ir = True
    x = ti.var(ti.i32)

    p, q = 3, 7
    n, m = 2, 4

    @ti.layout
    def place():
        ti.root.dense(ti.ij, (p, q)).dense(ti.ij, (n, m)).place(x)

    @ti.kernel
    def iterate():
        for i, j in x.parent():
            x[i, j] += 1

    iterate()

    for i in range(p):
        for j in range(q):
            assert x[i * n, j * m] == 1
