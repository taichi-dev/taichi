import taichi as ti


@ti.all_archs
def test_normal_grad():
    x = ti.var(ti.f32)
    loss = ti.var(ti.f32)

    n = 128

    @ti.layout
    def place():
        ti.root.dense(ti.i, n).place(x)
        ti.root.place(loss)
        ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(n):
            ti.atomic_add(loss, x[i]**2)

    for i in range(n):
        x[i] = i

    with ti.Tape(loss):
        func()

    for i in range(n):
        assert x.grad[i] == i * 2


@ti.all_archs
def test_stop_grad():
    x = ti.var(ti.f32)
    loss = ti.var(ti.f32)

    n = 128

    @ti.layout
    def place():
        ti.root.dense(ti.i, n).place(x)
        ti.root.place(loss)
        ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(n):
            ti.core.stop_grad(x.snode().ptr)
            ti.atomic_add(loss, x[i]**2)

    for i in range(n):
        x[i] = i

    with ti.Tape(loss):
        func()

    for i in range(n):
        assert x.grad[i] == 0


@ti.all_archs
def test_stop_grad2():
    x = ti.var(ti.f32)
    loss = ti.var(ti.f32)

    n = 128

    @ti.layout
    def place():
        ti.root.dense(ti.i, n).place(x)
        ti.root.place(loss)
        ti.root.lazy_grad()

    @ti.kernel
    def func():
        # Two loops, one with stop grad on without
        for i in range(n):
            ti.stop_grad(x)
            ti.atomic_add(loss, x[i]**2)
        for i in range(n):
            ti.atomic_add(loss, x[i]**2)

    for i in range(n):
        x[i] = i

    with ti.Tape(loss):
        func()

    # If without stop, grad x.grad[i] = i * 4
    for i in range(n):
        assert x.grad[i] == i * 2
