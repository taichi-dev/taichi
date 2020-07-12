import taichi as ti


@ti.archs_support_sparse
def test_pointer():
    x = ti.var(ti.f32)
    s = ti.var(ti.i32)

    n = 16

    ptr = ti.root.pointer(ti.i, n)
    ptr.dense(ti.i, n).place(x)
    ti.root.place(s)

    s[None] = 0

    @ti.kernel
    def activate():
        ti.activate(ptr, 1)
        ti.activate(ptr, 32)

    @ti.kernel
    def func():
        for i in x:
            s[None] += 1

    activate()
    func()
    assert s[None] == 32


test_pointer()
