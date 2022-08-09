import taichi as ti
from tests import test_utils


@test_utils.test()
def test_abs():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)

    N = 16

    ti.root.dense(ti.i, N).place(x)
    ti.root.dense(ti.i, N).place(y)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(N):
            x[i] = abs(y[i])

    for i in range(N):
        y[i] = i - 10
        x.grad[i] = 1

    func()
    func.grad()

    def sgn(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    for i in range(N):
        assert x[i] == abs(y[i])
        assert y.grad[i] == sgn(y[i])


@test_utils.test()
def test_abs_fwd():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)
    N = 16

    ti.root.dense(ti.i, N).place(x)
    ti.root.dense(ti.i, N).place(y)
    ti.root.lazy_dual()

    @ti.kernel
    def func():
        for i in range(N):
            x[i] = abs(y[i])

    for i in range(N):
        y[i] = i - 10

    with ti.ad.FwdMode(loss=x, param=y, seed=[1.0 for _ in range(N)]):
        func()

    def sgn(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    for i in range(N):
        assert x[i] == abs(y[i])
        assert x.dual[i] == sgn(y[i])
