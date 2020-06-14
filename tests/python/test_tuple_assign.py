import taichi as ti

@ti.all_archs
def test_fibonacci():
    @ti.kernel
    def ti_fibonacci(n: ti.i32) -> ti.i32:
        a, b = 0, 1
        if 1:
            for i in range(n):
                a, b = b, a + b
        return b

    def py_fibonacci(n):
        a, b = 0, 1
        for i in range(n):
            a, b = b, a + b
        return b

    for n in range(5):
        assert ti_fibonacci(n) == py_fibonacci(n)

@ti.host_arch_only
def test_swap2():
    a = ti.var(ti.f32, ())
    b = ti.var(ti.f32, ())

    @ti.kernel
    def func():
        a[None], b[None] = b[None], a[None]

    a[None] = 2
    b[None] = 3
    func()
    assert a[None] == 3
    assert b[None] == 2

@ti.host_arch_only
def test_assign2_static():
    ti.init(print_preprocessed=True)
    a = ti.var(ti.f32, ())
    b = ti.var(ti.f32, ())

    @ti.kernel
    def func():
        # XXX: why a, b = ti.static(b, a) doesn't work?
        c, d = ti.static(b, a)
        c[None], d[None] = 2, 3

    func()
    assert a[None] == 3
    assert b[None] == 2

@ti.host_arch_only
def test_swap3():
    a = ti.var(ti.f32, ())
    b = ti.var(ti.f32, ())
    c = ti.var(ti.f32, ())

    @ti.kernel
    def func():
        a[None], b[None], c[None] = b[None], c[None], a[None]

    a[None] = 2
    b[None] = 3
    c[None] = 4
    func()
    assert a[None] == 3
    assert b[None] == 4
    assert c[None] == 2
