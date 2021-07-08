import pytest

import taichi as ti


@ti.all_archs
def test_POT():
    val = ti.field(ti.i32)

    n = 4
    m = 8
    p = 16

    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

    assert val.shape == (n, m, p)
    assert val.dtype == ti.i32


@ti.all_archs
def test_non_POT():
    val = ti.field(ti.i32)

    n = 3
    m = 7
    p = 11

    blk1 = ti.root.dense(ti.i, n)
    blk2 = blk1.dense(ti.j, m)
    blk3 = blk2.dense(ti.k, p)
    blk3.place(val)

    assert val.shape == (n, m, p)
    assert val.dtype == ti.i32


@ti.all_archs
def test_unordered():
    val = ti.field(ti.i32)

    n = 3
    m = 7
    p = 11

    blk1 = ti.root.dense(ti.k, n)
    blk2 = blk1.dense(ti.i, m)
    blk3 = blk2.dense(ti.j, p)
    blk3.place(val)

    assert val.dtype == ti.i32
    assert val.shape == (n, m, p)
    assert val.snode.parent(0) == val.snode
    assert val.snode.parent() == blk3
    assert val.snode.parent(1) == blk3
    assert val.snode.parent(2) == blk2
    assert val.snode.parent(3) == blk1
    assert val.snode.parent(4) == ti.root

    assert val.snode in blk3.get_children()
    assert blk3 in blk2.get_children()
    assert blk2 in blk1.get_children()
    assert blk1 in ti.FieldsBuilder.finalized_fbs()[0].root.get_children()

    expected_str = f'ti.root => dense {[n]} => dense {[n, m]}' \
        f' => dense {[n, m, p]} => place {[n, m, p]}'
    assert str(val.snode) == expected_str


@ti.all_archs
def test_unordered_matrix():
    val = ti.Matrix.field(3, 2, ti.i32)

    n = 3
    m = 7
    p = 11

    blk1 = ti.root.dense(ti.k, n)
    blk2 = blk1.dense(ti.i, m)
    blk3 = blk2.dense(ti.j, p)
    blk3.place(val)

    assert val.shape == (n, m, p)
    assert val.dtype == ti.i32
    assert val.loop_range().snode.parent(0) == val.loop_range().snode
    assert val.loop_range().snode.parent() == blk3
    assert val.loop_range().snode.parent(1) == blk3
    assert val.loop_range().snode.parent(2) == blk2
    assert val.loop_range().snode.parent(3) == blk1
    assert val.loop_range().snode.parent(4) == ti.root


@pytest.mark.filterwarnings('ignore')
@ti.host_arch_only
def test_deprecated():
    val = ti.field(ti.f32)
    mat = ti.Matrix.field(3, 2, ti.i32)

    n = 3
    m = 7
    p = 11

    blk1 = ti.root.dense(ti.k, n)
    blk2 = blk1.dense(ti.i, m)
    blk3 = blk2.dense(ti.j, p)
    blk3.place(val, mat)

    assert val.dim() == 3
    assert val.data_type() == ti.f32
    assert val.shape() == (n, m, p)
    assert mat.dim() == 3
    assert mat.data_type() == ti.i32
    assert mat.shape() == (n, m, p)
    assert blk3.dim() == 3
    assert blk3.shape() == (n, m, p)
    assert val.snode().parent() == blk3
    assert mat.snode().parent() == blk3


@ti.all_archs
def test_parent_exceeded():
    val = ti.field(ti.f32)

    m = 7
    n = 3

    blk1 = ti.root.dense(ti.i, m)
    blk2 = blk1.dense(ti.j, n)
    blk2.place(val)

    assert val.snode.parent() == blk2
    assert val.snode.parent(2) == blk1
    assert val.snode.parent(3) == ti.root
    assert val.snode.parent(4) == None
    assert val.snode.parent(42) == None

    assert ti.root.parent() == None
