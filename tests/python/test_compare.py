import taichi as ti


@ti.require(ti.extension.sparse)
@ti.all_archs
def test_compare_basics():
    a = ti.field(ti.i32)
    ti.root.dynamic(ti.i, 256).place(a)
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        b[None] = 3
        c[None] = 5
        a[0] = b < c
        a[1] = b <= c
        a[2] = b > c
        a[3] = b >= c
        a[4] = b == c
        a[5] = b != c
        a[6] = c < b
        a[7] = c <= b
        a[8] = c > b
        a[9] = c >= b
        a[10] = c == b
        a[11] = c != b

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


@ti.require(ti.extension.sparse)
@ti.all_archs
def test_compare_equality():
    a = ti.field(ti.i32)
    ti.root.dynamic(ti.i, 256).place(a)
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        b[None] = 3
        c[None] = 3
        a[0] = b < c
        a[1] = b <= c
        a[2] = b > c
        a[3] = b >= c
        a[4] = b == c
        a[5] = b != c
        a[6] = c < b
        a[7] = c <= b
        a[8] = c > b
        a[9] = c >= b
        a[10] = c == b
        a[11] = c != b

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


@ti.require(ti.extension.sparse)
@ti.all_archs
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


@ti.all_archs
def test_no_duplicate_eval_func():
    a = ti.field(ti.i32, ())
    b = ti.field(ti.i32, ())

    @ti.func
    def why_this_foo_fail(n):
        return ti.atomic_add(b[None], n)

    def foo(n):
        return ti.atomic_add(ti.subscript(b, None), n)

    @ti.kernel
    def func():
        a[None] = 0 <= foo(2) < 1

    func()
    assert a[None] == 1
    assert b[None] == 2


@ti.require(ti.extension.sparse)
@ti.all_archs
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
        a[0] = c == d != b < d > b >= b <= c
        a[1] = b <= c != d > b == b

    func()
    assert a[0]
    assert not a[1]
