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
    def wrapper(*args, **kwargs):
        if has_autograd:
            func(*args, **kwargs)

    return wrapper


# Note: test happens at v = 0.2
def grad_test(tifunc, npfunc=None, default_fp=ti.f32):
    if npfunc is None:
        npfunc = tifunc

    @ti.all_archs_with(default_fp=default_fp)
    def impl():
        print(f'arch={ti.cfg.arch} default_fp={ti.cfg.default_fp}')
        x = ti.field(default_fp)
        y = ti.field(default_fp)

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

        assert y[0] == approx(npfunc(v))
        assert x.grad[0] == approx(grad(npfunc)(v))

    impl()


@if_has_autograd
@ti.all_archs
def test_size1():
    x = ti.field(ti.i32)

    ti.root.dense(ti.i, 1).place(x)

    x[0] = 1
    assert x[0] == 1


@if_has_autograd
def test_poly():
    grad_test(lambda x: x)
    grad_test(lambda x: -x)
    grad_test(lambda x: x * x)
    grad_test(lambda x: x**2)
    grad_test(lambda x: x * x * x)
    grad_test(lambda x: x * x * x * x)
    grad_test(lambda x: 0.4 * x * x - 3)
    grad_test(lambda x: (x - 3) * (x - 1))
    grad_test(lambda x: (x - 3) * (x - 1) + x * x)


@if_has_autograd
def test_trigonometric():
    grad_test(lambda x: ti.tanh(x), lambda x: np.tanh(x))
    grad_test(lambda x: ti.sin(x), lambda x: np.sin(x))
    grad_test(lambda x: ti.cos(x), lambda x: np.cos(x))
    grad_test(lambda x: ti.acos(x), lambda x: np.arccos(x))
    grad_test(lambda x: ti.asin(x), lambda x: np.arcsin(x))


@if_has_autograd
def test_frac():
    grad_test(lambda x: 1 / x)
    grad_test(lambda x: (x + 1) / (x - 1))
    grad_test(lambda x: (x + 1) * (x + 2) / ((x - 1) * (x + 3)))


@if_has_autograd
def test_unary():
    grad_test(lambda x: ti.sqrt(x), lambda x: np.sqrt(x))
    grad_test(lambda x: ti.exp(x), lambda x: np.exp(x))
    grad_test(lambda x: ti.log(x), lambda x: np.log(x))


@if_has_autograd
def test_minmax():
    grad_test(lambda x: ti.min(x, 0), lambda x: np.minimum(x, 0))
    grad_test(lambda x: ti.min(x, 1), lambda x: np.minimum(x, 1))
    grad_test(lambda x: ti.min(0, x), lambda x: np.minimum(0, x))
    grad_test(lambda x: ti.min(1, x), lambda x: np.minimum(1, x))

    grad_test(lambda x: ti.max(x, 0), lambda x: np.maximum(x, 0))
    grad_test(lambda x: ti.max(x, 1), lambda x: np.maximum(x, 1))
    grad_test(lambda x: ti.max(0, x), lambda x: np.maximum(0, x))
    grad_test(lambda x: ti.max(1, x), lambda x: np.maximum(1, x))


@if_has_autograd
@ti.all_archs
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


@if_has_autograd
def test_atan2():
    grad_test(lambda x: ti.atan2(0.4, x), lambda x: np.arctan2(0.4, x))
    grad_test(lambda y: ti.atan2(y, 0.4), lambda y: np.arctan2(y, 0.4))


@if_has_autograd
def test_atan2_f64():
    grad_test(lambda x: ti.atan2(0.4, x),
              lambda x: np.arctan2(0.4, x),
              default_fp=ti.f64)
    grad_test(lambda y: ti.atan2(y, 0.4),
              lambda y: np.arctan2(y, 0.4),
              default_fp=ti.f64)


@if_has_autograd
def test_pow():
    grad_test(lambda x: 0.4**x, lambda x: np.power(0.4, x))
    grad_test(lambda y: y**0.4, lambda y: np.power(y, 0.4))


@if_has_autograd
def test_pow_f64():
    grad_test(lambda x: 0.4**x, lambda x: np.power(0.4, x), default_fp=ti.f64)
    grad_test(lambda y: y**0.4, lambda y: np.power(y, 0.4), default_fp=ti.f64)


@ti.all_archs
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


@ti.all_archs
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


@ti.all_archs
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


@ti.require(ti.extension.data64)
@ti.all_archs
def test_cast():
    @ti.kernel
    def func():
        print(ti.cast(ti.cast(ti.cast(1.0, ti.f64), ti.f32), ti.f64))

    func()


@ti.require(ti.extension.data64)
@ti.all_archs
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


@ti.require(ti.extension.data64)
@ti.all_archs
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
