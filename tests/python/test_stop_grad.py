import taichi as ti
from tests import test_utils


@test_utils.test()
def test_normal_grad():
    x = ti.field(ti.f32)
    loss = ti.field(ti.f32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.place(loss)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(n):
            loss[None] += x[i]**2

    for i in range(n):
        x[i] = i

    with ti.Tape(loss):
        func()

    for i in range(n):
        assert x.grad[i] == i * 2


@test_utils.test()
def test_stop_grad():
    x = ti.field(ti.f32)
    loss = ti.field(ti.f32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.place(loss)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(n):
            ti.stop_grad(x)
            loss[None] += x[i]**2

    for i in range(n):
        x[i] = i

    with ti.Tape(loss):
        func()

    for i in range(n):
        assert x.grad[i] == 0


@test_utils.test()
def test_stop_grad2():
    x = ti.field(ti.f32)
    loss = ti.field(ti.f32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.place(loss)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        # Two loops, one with stop grad on without
        for i in range(n):
            ti.stop_grad(x)
            loss[None] += x[i]**2
        for i in range(n):
            loss[None] += x[i]**2

    for i in range(n):
        x[i] = i

    with ti.Tape(loss):
        func()

    # If without stop, grad x.grad[i] = i * 4
    for i in range(n):
        assert x.grad[i] == i * 2
