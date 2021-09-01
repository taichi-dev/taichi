import taichi as ti


def _test_nested():
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


# TODO: remove excluding of ti.metal.
@ti.test(require=ti.extension.sparse,
         exclude=[ti.metal],
         demote_dense_struct_fors=False,
         packed=False)
def test_nested():
    _test_nested()


@ti.test(demote_dense_struct_fors=True, packed=False)
def test_nested_demote():
    _test_nested()


@ti.test(require=[ti.extension.sparse, ti.extension.packed],
         demote_dense_struct_fors=False,
         packed=True)
def test_nested_packed():
    _test_nested()


@ti.test(require=ti.extension.packed,
         demote_dense_struct_fors=True,
         packed=True)
def test_nested_demote_packed():
    _test_nested()
