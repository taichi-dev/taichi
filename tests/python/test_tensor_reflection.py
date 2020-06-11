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
    assert val.data_type() == ti.i32


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
    assert val.data_type() == ti.i32


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

    assert val.data_type() == ti.i32
    assert val.shape() == (n, m, p)
    assert val.dim() == 3
    assert val.snode().parent().ptr == blk3.ptr
    assert val.snode().parent(1).ptr == blk3.ptr
    assert val.snode().parent(2).ptr == blk2.ptr
    assert val.snode().parent(3).ptr == blk1.ptr


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
    assert val.data_type() == ti.i32
    assert val.snode().parent(0).ptr == val.snode().ptr
    assert val.snode().parent().ptr == blk3.ptr
    assert val.snode().parent(1).ptr == blk3.ptr
    assert val.snode().parent(2).ptr == blk2.ptr
    assert val.snode().parent(3).ptr == blk1.ptr


@ti.all_archs
def _test_var_parent():  # doesn't work :(
    val1 = ti.Matrix(3, 2, ti.i32)
    val2 = ti.Matrix(3, 2, ti.i32)
    val3 = ti.Matrix(3, 2, ti.i32)

    n = 3
    m = 7
    p = 11

    blk1 = ti.root.dense(ti.k, n).place(val1)
    blk2 = blk1.dense(ti.i, m).place(val2)
    blk3 = blk2.dense(ti.j, p).place(val3)

    assert val3.parent().ptr == val3.ptr
    assert val3.parent(1).ptr == val3.ptr
    assert val3.parent(2).ptr == val2.ptr
    assert val3.parent(3).ptr == val1.ptr
