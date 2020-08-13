import taichi as ti


@ti.all_archs
def test_running_loss():
    return
    steps = 16

    total_loss = ti.field(ti.f32)
    running_loss = ti.field(ti.f32)
    additional_loss = ti.field(ti.f32)

    ti.root.place(total_loss)
    ti.root.dense(ti.i, steps).place(running_loss)
    ti.root.place(additional_loss)
    ti.root.lazy_grad()

    @ti.kernel
    def compute_loss():
        total_loss[None] = 0.0
        for i in range(steps):
            total_loss[None].atomic_add(running_loss[i] * 2)
        total_loss[None].atomic_add(additional_loss[None] * 3)

    compute_loss()

    assert total_loss.grad[None] == 1
    for i in range(steps):
        assert running_loss[i] == 2
    assert additional_loss.grad[None] == 3


@ti.all_archs
def test_reduce_separate():
    a = ti.field(ti.f32, shape=(16))
    b = ti.field(ti.f32, shape=(4))
    c = ti.field(ti.f32, shape=())

    ti.root.lazy_grad()

    @ti.kernel
    def reduce1():
        for i in range(16):
            b[i // 4] += a[i]

    @ti.kernel
    def reduce2():
        for i in range(4):
            c[None] += b[i]

    c.grad[None] = 1
    reduce2.grad()
    reduce1.grad()

    for i in range(4):
        assert b.grad[i] == 1
    for i in range(16):
        assert a.grad[i] == 1


@ti.all_archs
def test_reduce_merged():
    a = ti.field(ti.f32, shape=(16))
    b = ti.field(ti.f32, shape=(4))
    c = ti.field(ti.f32, shape=())

    ti.root.lazy_grad()

    @ti.kernel
    def reduce():
        for i in range(16):
            b[i // 4] += a[i]

        for i in range(4):
            c[None] += b[i]

    c.grad[None] = 1
    reduce.grad()

    for i in range(4):
        assert b.grad[i] == 1
    for i in range(16):
        assert a.grad[i] == 1
