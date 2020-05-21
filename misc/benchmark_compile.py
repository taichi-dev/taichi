import taichi as ti
from pytest import approx
import autograd.numpy as np
from autograd import grad


@ti.all_archs
def grad_test(tifunc, npfunc=None):
    if npfunc is None:
        npfunc = tifunc

    x = ti.var(ti.f32)
    y = ti.var(ti.f32)

    @ti.layout
    def place():
        ti.root.dense(ti.i, 1).place(x, x.grad, y, y.grad)

    @ti.kernel
    def func():
        for i in x:
            y[i] = tifunc(x[i])

    v = 0.2

    y.grad[0] = 1
    x[0] = v
    func()
    func.grad()

    assert y[0] == approx(npfunc(v))
    assert x.grad[0] == approx(grad(npfunc)(v))


def test_unary():
    import time
    t = time.time()
    grad_test(lambda x: ti.sqrt(x), lambda x: np.sqrt(x))
    grad_test(lambda x: ti.exp(x), lambda x: np.exp(x))
    grad_test(lambda x: ti.log(x), lambda x: np.log(x))
    ti.core.print_profile_info()
    print("Total time {:.3f}s".format(time.time() - t))


test_unary()
