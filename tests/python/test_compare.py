import pytest
from taichi.lang import impl

import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.sparse)
def test_compare_basics():
    a = ti.field(ti.i32)
    ti.root.dynamic(ti.i, 256).place(a)
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        b[None] = 3
        c[None] = 5
        a[0] = b[None] < c[None]
        a[1] = b[None] <= c[None]
        a[2] = b[None] > c[None]
        a[3] = b[None] >= c[None]
        a[4] = b[None] == c[None]
        a[5] = b[None] != c[None]
        a[6] = c[None] < b[None]
        a[7] = c[None] <= b[None]
        a[8] = c[None] > b[None]
        a[9] = c[None] >= b[None]
        a[10] = c[None] == b[None]
        a[11] = c[None] != b[None]

    func()
    assert a[0]
    assert a[1]
    assert not a[2]
    assert not a[3]
    assert not a[4]
    assert a[5]
    assert not a[6]
    assert not a[7]
    assert a[8]
    assert a[9]
    assert not a[10]
    assert a[11]


@test_utils.test(require=ti.extension.sparse)
def test_compare_equality():
    a = ti.field(ti.i32)
    ti.root.dynamic(ti.i, 256).place(a)
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        b[None] = 3
        c[None] = 3
        a[0] = b[None] < c[None]
        a[1] = b[None] <= c[None]
        a[2] = b[None] > c[None]
        a[3] = b[None] >= c[None]
        a[4] = b[None] == c[None]
        a[5] = b[None] != c[None]
        a[6] = c[None] < b[None]
        a[7] = c[None] <= b[None]
        a[8] = c[None] > b[None]
        a[9] = c[None] >= b[None]
        a[10] = c[None] == b[None]
        a[11] = c[None] != b[None]

    func()
    assert not a[0]
    assert a[1]
    assert not a[2]
    assert a[3]
    assert a[4]
    assert not a[5]
    assert not a[6]
    assert a[7]
    assert not a[8]
    assert a[9]
    assert a[10]
    assert not a[11]


@test_utils.test(require=ti.extension.sparse)
def test_no_duplicate_eval():
    a = ti.field(ti.i32)
    ti.root.dynamic(ti.i, 256).place(a)

    @ti.kernel
    def func():
        a[2] = 0 <= ti.append(a.parent(), [], 10) < 1

    func()
    assert a[0] == 10
    assert a[1] == 0  # not appended twice
    assert a[2]  # ti.append returns 0


@test_utils.test()
def test_no_duplicate_eval_func():
    a = ti.field(ti.i32, ())
    b = ti.field(ti.i32, ())

    @ti.func
    def why_this_foo_fail(n):
        return ti.atomic_add(b[None], n)

    def foo(n):
        return ti.atomic_add(impl.subscript(b, None), n)

    @ti.kernel
    def func():
        a[None] = 0 <= foo(2) < 1

    func()
    assert a[None] == 1
    assert b[None] == 2


@test_utils.test(require=ti.extension.sparse)
def test_chain_compare():
    a = ti.field(ti.i32)
    ti.root.dynamic(ti.i, 256).place(a)
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())
    d = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        b[None] = 2
        c[None] = 3
        d[None] = 3
        a[0] = c[None] == d[None] != b[None] < d[None] > b[None] >= b[
            None] <= c[None]
        a[1] = b[None] <= c[None] != d[None] > b[None] == b[None]

    func()
    assert a[0]
    assert not a[1]


@test_utils.test()
def test_static_in():
    @ti.kernel
    def foo(a: ti.template()) -> ti.i32:
        b = 0
        if ti.static(a in [ti.i32, ti.u32]):
            b = 1
        elif ti.static(a not in [ti.f32, ti.f64]):
            b = 2
        return b

    assert foo(ti.u32) == 1
    assert foo(ti.i64) == 2
    assert foo(ti.f32) == 0


@test_utils.test()
def test_non_static_in():
    with pytest.raises(ti.TaichiCompilationError,
                       match='"In" is only supported inside `ti.static`.'):

        @ti.kernel
        def foo(a: ti.template()) -> ti.i32:
            b = 0
            if a in [ti.i32, ti.u32]:
                b = 1
            return b

        foo(ti.i32)
