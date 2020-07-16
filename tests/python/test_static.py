import taichi as ti
import numpy as np


@ti.all_archs
def test_static_if():
    for val in [0, 1]:
        ti.init()
        x = ti.var(ti.i32)

        ti.root.dense(ti.i, 1).place(x)

        @ti.kernel
        def static():
            if ti.static(val > 0.5):
                x[0] = 1
            else:
                x[0] = 0

        static()
        assert x[0] == val


@ti.must_throw(AssertionError)
def test_static_if_error():
    x = ti.var(ti.i32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def static():
        if ti.static(x[0]):
            x[0] = 1
        else:
            x[0] = 0

    static()


@ti.all_archs
def test_static_ndrange():
    n = 3
    x = ti.Matrix(n, n, dt=ti.f32, shape=(n, n))

    @ti.kernel
    def fill():
        w = [0, 1, 2]
        for i, j in ti.static(ti.ndrange(3, 3)):
            x[i, j][i, j] = w[i] + w[j] * 2

    fill()
    for i in range(3):
        for j in range(3):
            assert x[i, j][i, j] == i + j * 2


@ti.host_arch_only
def test_static_break():
    x = ti.var(ti.i32, 5)

    @ti.kernel
    def func():
        for i in ti.static(range(5)):
            x[i] = 1
            if ti.static(i == 2):
                break

    func()

    assert np.allclose(x.to_numpy(), np.array([1, 1, 1, 0, 0]))


@ti.host_arch_only
def test_static_continue():
    x = ti.var(ti.i32, 5)

    @ti.kernel
    def func():
        for i in ti.static(range(5)):
            if ti.static(i == 2):
                continue
            x[i] = 1

    func()

    assert np.allclose(x.to_numpy(), np.array([1, 1, 0, 1, 1]))
