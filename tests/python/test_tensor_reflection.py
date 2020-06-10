import taichi as ti


@ti.all_archs
def test_POT():
    val = ti.var(ti.i32)

    n = 4
    m = 8
    p = 16

    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

    assert val.shape() == (n, m, p)
    assert val.dim() == 3
    assert val.data_type() == ti.f32


@ti.all_archs
def test_non_POT():
    val = ti.var(ti.i32)

    n = 3
    m = 7
    p = 11

    blk1 = ti.root.dense(ti.i, n)
    blk2 = blk1.dense(ti.j, m)
    blk3 = blk2.dense(ti.k, p)
    blk3.place(val)

    assert val.shape() == (n, m, p)
    assert val.dim() == 3
    assert val.data_type() == ti.f32
    assert val.snode() == blk3
    assert val.snode().parent() == blk2
    assert val.parent(0) == blk3
    assert val.parent() == blk2
    assert val.parent(2) == blk1


@ti.all_archs
def test_unordered():
    val = ti.var(ti.i32)

    n = 3
    m = 7
    p = 11

    blk1 = ti.root.dense(ti.k, n)
    blk2 = blk1.dense(ti.i, m)
    blk3 = blk2.dense(ti.j, p)
    blk3.place(val)

    assert val.data_type() == ti.f32
    assert val.shape() == (n, m, p)
    assert val.dim() == 3
    assert val.snode() == blk3
    assert val.snode().parent() == blk2
    assert val.parent(0) == blk3
    assert val.parent() == blk2
    assert val.parent(2) == blk1


@ti.all_archs
def test_unordered_matrix():
    val = ti.Matrix(3, 2, ti.i32)

    n = 3
    m = 7
    p = 11

    blk1 = ti.root.dense(ti.k, n)
    blk2 = blk1.dense(ti.i, m)
    blk3 = blk2.dense(ti.j, p)
    blk3.place(val)

    assert val.dim() == 3
    assert val.shape() == (n, m, p)
    assert val.data_type() == ti.f32
    assert val.snode() == blk3
    assert val.snode().parent() == blk2
    assert val.parent(0) == blk3
    assert val.parent() == blk2
    assert val.parent(2) == blk1
