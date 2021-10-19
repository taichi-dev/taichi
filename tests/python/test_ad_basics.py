import functools

import pytest

import taichi as ti
from taichi import approx

has_autograd = False

try:
    import autograd.numpy as np
    from autograd import grad
    has_autograd = True
except:
    pass


def if_has_autograd(func):
    # functools.wraps is nececssary for pytest parametrization to work
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if has_autograd:
            func(*args, **kwargs)

    return wrapper


# Note: test happens at v = 0.2
def grad_test(tifunc, npfunc=None):
    npfunc = npfunc or tifunc

    print(f'arch={ti.cfg.arch} default_fp={ti.cfg.default_fp}')
    x = ti.field(ti.cfg.default_fp)
    y = ti.field(ti.cfg.default_fp)

    ti.root.dense(ti.i, 1).place(x, x.grad, y, y.grad)

    @ti.kernel
    def func():
        for i in x:
            y[i] = tifunc(x[i])

    v = 0.234

    y.grad[0] = 1
    x[0] = v
    func()
    func.grad()

    assert y[0] == approx(npfunc(v), rel=1e-4)
    assert x.grad[0] == approx(grad(npfunc)(v), rel=1e-4)


@if_has_autograd
@ti.test()
def test_size1():
    x = ti.field(ti.i32)

    ti.root.dense(ti.i, 1).place(x)

    x[0] = 1
    assert x[0] == 1


@pytest.mark.parametrize('tifunc', [
    lambda x: x,
    lambda x: -x,
    lambda x: x * x,
    lambda x: x**2,
    lambda x: x * x * x,
    lambda x: x * x * x * x,
    lambda x: 0.4 * x * x - 3,
    lambda x: (x - 3) * (x - 1),
    lambda x: (x - 3) * (x - 1) + x * x,
])
@if_has_autograd
@ti.test()
def test_poly(tifunc):
    grad_test(tifunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: ti.tanh(x), lambda x: np.tanh(x)),
    (lambda x: ti.sin(x), lambda x: np.sin(x)),
    (lambda x: ti.cos(x), lambda x: np.cos(x)),
    (lambda x: ti.acos(x), lambda x: np.arccos(x)),
    (lambda x: ti.asin(x), lambda x: np.arcsin(x)),
])
@if_has_autograd
@ti.test(exclude=[ti.vulkan])
def test_trigonometric(tifunc, npfunc):
    grad_test(tifunc, npfunc)


@pytest.mark.parametrize('tifunc', [
    lambda x: 1 / x,
    lambda x: (x + 1) / (x - 1),
    lambda x: (x + 1) * (x + 2) / ((x - 1) * (x + 3)),
])
@if_has_autograd
@ti.test()
def test_frac(tifunc):
    grad_test(tifunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: ti.sqrt(x), lambda x: np.sqrt(x)),
    (lambda x: ti.exp(x), lambda x: np.exp(x)),
    (lambda x: ti.log(x), lambda x: np.log(x)),
])
@if_has_autograd
@ti.test()
def test_unary(tifunc, npfunc):
    grad_test(tifunc, npfunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: ti.min(x, 0), lambda x: np.minimum(x, 0)),
    (lambda x: ti.min(x, 1), lambda x: np.minimum(x, 1)),
    (lambda x: ti.min(0, x), lambda x: np.minimum(0, x)),
    (lambda x: ti.min(1, x), lambda x: np.minimum(1, x)),
    (lambda x: ti.max(x, 0), lambda x: np.maximum(x, 0)),
    (lambda x: ti.max(x, 1), lambda x: np.maximum(x, 1)),
    (lambda x: ti.max(0, x), lambda x: np.maximum(0, x)),
    (lambda x: ti.max(1, x), lambda x: np.maximum(1, x)),
])
@if_has_autograd
@ti.test()
def test_minmax(tifunc, npfunc):
    grad_test(tifunc, npfunc)


@if_has_autograd
@ti.test()
def test_mod():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    ti.root.dense(ti.i, 1).place(x, y)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        y[0] = x[0] % 3

    @ti.kernel
    def func2():
        ti.atomic_add(y[0], x[0] % 3)

    func()
    func.grad()

    func2()
    func2.grad()


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: ti.atan2(0.4, x), lambda x: np.arctan2(0.4, x)),
    (lambda y: ti.atan2(y, 0.4), lambda y: np.arctan2(y, 0.4)),
])
@if_has_autograd
@ti.test()
def test_atan2(tifunc, npfunc):
    grad_test(tifunc, npfunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: ti.atan2(0.4, x), lambda x: np.arctan2(0.4, x)),
    (lambda y: ti.atan2(y, 0.4), lambda y: np.arctan2(y, 0.4)),
])
@if_has_autograd
@ti.test(require=ti.extension.data64, default_fp=ti.f64)
def test_atan2_f64(tifunc, npfunc):
    grad_test(tifunc, npfunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: 0.4**x, lambda x: np.power(0.4, x)),
    (lambda y: y**0.4, lambda y: np.power(y, 0.4)),
])
@if_has_autograd
@ti.test()
def test_pow(tifunc, npfunc):
    grad_test(tifunc, npfunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: 0.4**x, lambda x: np.power(0.4, x)),
    (lambda y: y**0.4, lambda y: np.power(y, 0.4)),
])
@if_has_autograd
@ti.test(require=ti.extension.data64, default_fp=ti.f64)
def test_pow_f64(tifunc, npfunc):
    grad_test(tifunc, npfunc)


@ti.test()
def test_obey_kernel_simplicity():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x, y)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in x:
            # OK: nested for loop
            for j in ti.static(range(3)):
                # OK: a series of non-for-loop statements
                y[i] += x[i] * 42
                y[i] -= x[i] * 5

    y.grad[0] = 1.0
    x[0] = 0.1

    func()
    func.grad()
    assert x.grad[0] == approx((42 - 5) * 3)


@ti.test()
def test_violate_kernel_simplicity1():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x, y)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in x:
            y[i] = x[i] * 42
            for j in ti.static(range(3)):
                y[i] += x[i]

    func()
    func.grad()


@ti.test()
def test_violate_kernel_simplicity2():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x, y)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in x:
            for j in ti.static(range(3)):
                y[i] += x[i]
            y[i] += x[i] * 42

    func()
    func.grad()


@ti.test(require=ti.extension.data64)
def test_cast():
    @ti.kernel
    def func():
        print(ti.cast(ti.cast(ti.cast(1.0, ti.f64), ti.f32), ti.f64))

    func()


@ti.test(require=ti.extension.data64)
def test_ad_precision_1():
    loss = ti.field(ti.f32, shape=())
    x = ti.field(ti.f64, shape=())

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        loss[None] = x[None]

    loss.grad[None] = 1
    func.grad()

    assert x.grad[None] == 1


@ti.test(require=ti.extension.data64)
def test_ad_precision_2():
    loss = ti.field(ti.f64, shape=())
    x = ti.field(ti.f32, shape=())

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        loss[None] = x[None]

    with ti.Tape(loss):
        func()

    assert x.grad[None] == 1


@ti.test()
def test_ad_rand():
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def work():
        loss[None] = x[None] * ti.random()

    x[None] = 10
    with pytest.raises(RuntimeError) as e:
        with ti.Tape(loss):
            work()
    assert 'RandStmt not supported' in e.value.args[0]
