import functools

import numpy as np
import pytest

import taichi as ti
from tests import test_utils

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

    print(
        f'arch={ti.lang.impl.current_cfg().arch} default_fp={ti.lang.impl.current_cfg().default_fp}'
    )
    x = ti.field(ti.lang.impl.current_cfg().default_fp)
    y = ti.field(ti.lang.impl.current_cfg().default_fp)

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

    assert y[0] == test_utils.approx(npfunc(v), rel=1e-4)
    assert x.grad[0] == test_utils.approx(grad(npfunc)(v), rel=1e-4)


def grad_test_fwd(tifunc, npfunc=None):
    npfunc = npfunc or tifunc

    print(
        f'arch={ti.lang.impl.current_cfg().arch} default_fp={ti.lang.impl.current_cfg().default_fp}'
    )
    x = ti.field(ti.lang.impl.current_cfg().default_fp)
    y = ti.field(ti.lang.impl.current_cfg().default_fp)

    ti.root.dense(ti.i, 1).place(x, x.dual, y, y.dual)

    @ti.kernel
    def func():
        for i in x:
            y[i] = tifunc(x[i])

    v = 0.234

    x[0] = v
    with ti.ad.FwdMode(loss=y, param=x, seed=[1.0]):
        func()

    assert y[0] == test_utils.approx(npfunc(v), rel=1e-4)
    assert y.dual[0] == test_utils.approx(grad(npfunc)(v), rel=1e-4)


@if_has_autograd
@test_utils.test()
def test_size1():
    x = ti.field(ti.i32)

    ti.root.dense(ti.i, 1).place(x)

    x[0] = 1
    assert x[0] == 1


@pytest.mark.parametrize('tifunc', [
    lambda x: x,
    lambda x: ti.abs(-x),
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
@test_utils.test(exclude=[ti.cc])
def test_poly(tifunc):
    grad_test(tifunc)
    grad_test_fwd(tifunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: ti.tanh(x), lambda x: np.tanh(x)),
    (lambda x: ti.sin(x), lambda x: np.sin(x)),
    (lambda x: ti.cos(x), lambda x: np.cos(x)),
    (lambda x: ti.acos(x), lambda x: np.arccos(x)),
    (lambda x: ti.asin(x), lambda x: np.arcsin(x)),
])
@if_has_autograd
@test_utils.test(exclude=[ti.cc])
def test_trigonometric(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@pytest.mark.parametrize('tifunc', [
    lambda x: 1 / x,
    lambda x: (x + 1) / (x - 1),
    lambda x: (x + 1) * (x + 2) / ((x - 1) * (x + 3)),
])
@if_has_autograd
@test_utils.test(exclude=[ti.cc])
def test_frac(tifunc):
    grad_test(tifunc)
    grad_test_fwd(tifunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: ti.sqrt(x), lambda x: np.sqrt(x)),
    (lambda x: ti.exp(x), lambda x: np.exp(x)),
    (lambda x: ti.log(x), lambda x: np.log(x)),
])
@if_has_autograd
@test_utils.test(exclude=[ti.cc])
def test_unary(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


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
@test_utils.test(exclude=[ti.cc])
def test_minmax(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@if_has_autograd
@test_utils.test()
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
@test_utils.test()
def test_mod_fwd():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x, y)
    ti.root.lazy_dual()

    @ti.kernel
    def func():
        y[0] = x[0] % 3

    @ti.kernel
    def func2():
        ti.atomic_add(y[0], x[0] % 3)

    with ti.ad.FwdMode(loss=y, param=x, seed=[1.0]):
        func()
        func2()


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: ti.atan2(0.4, x), lambda x: np.arctan2(0.4, x)),
    (lambda y: ti.atan2(y, 0.4), lambda y: np.arctan2(y, 0.4)),
])
@if_has_autograd
@test_utils.test(exclude=[ti.cc])
def test_atan2(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: ti.atan2(0.4, x), lambda x: np.arctan2(0.4, x)),
    (lambda y: ti.atan2(y, 0.4), lambda y: np.arctan2(y, 0.4)),
])
@if_has_autograd
@test_utils.test(require=ti.extension.data64,
                 default_fp=ti.f64,
                 exclude=[ti.cc])
def test_atan2_f64(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: 0.4**x, lambda x: np.power(0.4, x)),
    (lambda y: y**0.4, lambda y: np.power(y, 0.4)),
])
@if_has_autograd
@test_utils.test(exclude=[ti.cc])
def test_pow(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@pytest.mark.parametrize('tifunc,npfunc', [
    (lambda x: 0.4**x, lambda x: np.power(0.4, x)),
    (lambda y: y**0.4, lambda y: np.power(y, 0.4)),
])
@if_has_autograd
@test_utils.test(require=ti.extension.data64,
                 default_fp=ti.f64,
                 exclude=[ti.cc])
def test_pow_f64(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@test_utils.test()
def test_select():
    N = 5
    loss = ti.field(ti.f32, shape=N)
    x = ti.field(ti.f32, shape=N)
    y = ti.field(ti.f32, shape=N)
    ti.root.lazy_grad()

    for i in range(N):
        x[i] = i
        y[i] = -i
        loss.grad[i] = 1.0

    @ti.kernel
    def func():
        for i in range(N):
            loss[i] += ti.select(i % 2, x[i], y[i])

    func()
    func.grad()
    for i in range(N):
        if i % 2:
            assert loss[i] == i
        else:
            assert loss[i] == -i
        assert x.grad[i] == i % 2 * 1.0
        assert y.grad[i] == (not i % 2) * 1.0


@test_utils.test()
def test_select_fwd():
    N = 5
    loss = ti.field(ti.f32, shape=N)
    x = ti.field(ti.f32, shape=N)
    y = ti.field(ti.f32, shape=N)
    ti.root.lazy_dual()

    for i in range(N):
        x[i] = i
        y[i] = -i

    @ti.kernel
    def func():
        for i in range(N):
            loss[i] = ti.select(i % 2, x[i], y[i])

    with ti.ad.FwdMode(loss=loss, param=x, seed=[1.0 for _ in range(N)]):
        func()

    for i in range(N):
        if i % 2:
            assert loss[i] == i
        else:
            assert loss[i] == -i
        assert loss.dual[i] == i % 2 * 1.0

    with ti.ad.FwdMode(loss=loss, param=y, seed=[1.0 for _ in range(N)]):
        func()

    for i in range(N):
        if i % 2:
            assert loss[i] == i
        else:
            assert loss[i] == -i
        assert loss.dual[i] == (not i % 2) * 1.0


@test_utils.test()
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
    assert x.grad[0] == test_utils.approx((42 - 5) * 3)


@test_utils.test()
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


@test_utils.test()
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


@test_utils.test(require=ti.extension.data64)
def test_cast():
    @ti.kernel
    def func():
        print(ti.cast(ti.cast(ti.cast(1.0, ti.f64), ti.f32), ti.f64))

    func()


@test_utils.test(require=ti.extension.data64)
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


@test_utils.test(require=ti.extension.data64)
def test_ad_precision_2():
    loss = ti.field(ti.f64, shape=())
    x = ti.field(ti.f32, shape=())

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        loss[None] = x[None]

    with ti.ad.Tape(loss):
        func()

    assert x.grad[None] == 1


@test_utils.test()
def test_ad_rand():
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def work():
        loss[None] = x[None] * ti.random()

    x[None] = 10
    with pytest.raises(RuntimeError) as e:
        with ti.ad.Tape(loss):
            work()
    assert 'RandStmt not supported' in e.value.args[0]


@test_utils.test(exclude=[ti.cc, ti.vulkan, ti.opengl, ti.dx11])
def test_ad_frac():
    @ti.func
    def frac(x):
        fractional = x - ti.floor(x) if x > 0. else x - ti.ceil(x)
        return fractional

    @ti.kernel
    def ti_frac(input_field: ti.template(), output_field: ti.template()):
        for i in input_field:
            output_field[i] = frac(input_field[i])**2

    @ti.kernel
    def calc_loss(input_field: ti.template(), loss: ti.template()):
        for i in input_field:
            loss[None] += input_field[i]

    n = 10
    field0 = ti.field(dtype=ti.f32, shape=(n, ), needs_grad=True)
    randoms = np.random.randn(10).astype(np.float32)
    field0.from_numpy(randoms)
    field1 = ti.field(dtype=ti.f32, shape=(n, ), needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    with ti.ad.Tape(loss):
        ti_frac(field0, field1)
        calc_loss(field1, loss)

    grads = field0.grad.to_numpy()
    expected = np.modf(randoms)[0] * 2
    for i in range(n):
        assert grads[i] == test_utils.approx(expected[i], rel=1e-4)


@test_utils.test()
def test_ad_global_store_forwarding():
    x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    a = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    b = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    c = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    d = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    e = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def func():
        a[None] = x[None]
        b[None] = a[None] * 2
        c[None] = b[None] * 3
        d[None] = c[None] * 4
        e[None] = d[None] * 5

    x[None] = 1

    with ti.ad.Tape(loss=e):
        func()
    assert x.grad[None] == 120.0
    assert a.grad[None] == 120.0
    assert b.grad[None] == 60.0
    assert c.grad[None] == 20.0
    assert d.grad[None] == 5.0
