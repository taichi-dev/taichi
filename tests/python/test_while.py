import taichi as ti


@ti.all_archs
def test_while():
    x = ti.field(ti.f32)

    N = 1

    ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def func():
        i = 0
        s = 0
        while i < 10:
            s += i
            i += 1
        x[0] = s

    func()
    assert x[0] == 45


@ti.all_archs
def test_break():
    ret = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        i = 0
        s = 0
        while True:
            s += i
            i += 1
            if i > 10:
                break
        ret[None] = s

    func()
    assert ret[None] == 55
