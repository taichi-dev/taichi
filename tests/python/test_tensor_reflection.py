import pytest
from taichi.lang import impl

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_POT():
    val = ti.field(ti.i32)

    n = 4
    m = 8
    p = 16

    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

    assert val.shape == (n, m, p)
    assert val.dtype == ti.i32


@test_utils.test()
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


@test_utils.test()
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
    assert val.shape == (m, p, n)
    assert val.snode.parent(0) == val.snode
    assert val.snode.parent() == blk3
    assert val.snode.parent(1) == blk3
    assert val.snode.parent(2) == blk2
    assert val.snode.parent(3) == blk1
    assert val.snode.parent(4) == ti.root

    assert val.snode in blk3.get_children()
    assert blk3 in blk2.get_children()
    assert blk2 in blk1.get_children()
    impl.get_runtime().materialize_root_fb(False)
    assert blk1 in ti.FieldsBuilder._finalized_roots()[0].get_children()

    expected_str = f'ti.root => dense {[n]} => dense {[m, n]}' \
        f' => dense {[m, p, n]} => place {[m, p, n]}'
    assert str(val.snode) == expected_str


@test_utils.test()
def test_unordered_matrix():
    val = ti.Matrix.field(3, 2, ti.i32)

    n = 3
    m = 7
    p = 11

    blk1 = ti.root.dense(ti.k, n)
    blk2 = blk1.dense(ti.i, m)
    blk3 = blk2.dense(ti.j, p)
    blk3.place(val)

    assert val.shape == (m, p, n)
    assert val.dtype == ti.i32
    assert val.snode.parent(0) == val.snode
    assert val.snode.parent() == blk3
    assert val.snode.parent(1) == blk3
    assert val.snode.parent(2) == blk2
    assert val.snode.parent(3) == blk1
    assert val.snode.parent(4) == ti.root
    assert val.snode.path_from_root() == [ti.root, blk1, blk2, blk3, val.snode]


@test_utils.test()
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
