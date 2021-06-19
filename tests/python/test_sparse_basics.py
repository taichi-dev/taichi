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


@ti.test(require=ti.extension.sparse)
def test_pointer2():
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
