import pytest

import taichi as ti


@ti.test(require=ti.extension.sparse)
def test_pointer():
    x = ti.field(ti.f32)
    s = ti.field(ti.i32)

    n = 128

    ti.root.pointer(ti.i, n).dense(ti.i, n).place(x)
    ti.root.place(s)

    @ti.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[127] = 1
    x[256] = 1

    func()
    assert s[None] == 256


@ti.test(require=ti.extension.sparse)
def test_pointer_is_active():
    x = ti.field(ti.f32)
    s = ti.field(ti.i32)

    n = 128

    ti.root.pointer(ti.i, n).dense(ti.i, n).place(x)
    ti.root.place(s)

    @ti.kernel
    def func():
        for i in range(n * n):
            s[None] += ti.is_active(x.parent().parent(), i)

    x[0] = 1
    x[127] = 1
    x[256] = 1

    func()
    assert s[None] == 256


def _test_pointer2():
    x = ti.field(ti.f32)
    s = ti.field(ti.i32)

    n = 128

    ti.root.pointer(ti.i, n).pointer(ti.i, n).dense(ti.i, n).place(x)
    ti.root.place(s)

    @ti.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[127] = 1
    x[254] = 1
    x[256 + n * n] = 1

    x[257 + n * n] = 1
    x[257 + n * n * 2] = 1
    x[257 + n * n * 5] = 1

    func()
    assert s[None] == 5 * n
    print(x[257 + n * n * 7])
    assert s[None] == 5 * n


@ti.test(require=ti.extension.sparse)
def test_pointer2():
    _test_pointer2()


@ti.test(require=[ti.extension.sparse, ti.extension.packed], packed=True)
def test_pointer2_packed():
    _test_pointer2()


@pytest.mark.skip(reason='https://github.com/taichi-dev/taichi/issues/2520')
@ti.test(require=ti.extension.sparse, use_unified_memory=False)
def test_pointer_direct_place():
    x, y = ti.field(ti.i32), ti.field(ti.i32)

    N = 1
    ti.root.pointer(ti.i, N).place(x)
    ti.root.pointer(ti.i, N).place(y)

    @ti.kernel
    def foo():
        pass

    foo()
