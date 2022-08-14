import taichi as ti
from tests import test_utils


@test_utils.test()
def test_ad_fwd_add():
    N = 5
    x = ti.field(ti.f32, shape=N)
    loss = ti.field(ti.f32, shape=N)
    ti.root.lazy_dual()

    for i in range(N):
        x[i] = i

    @ti.kernel
    def ad_fwd_add():
        loss[1] += 2 * x[3]

    with ti.ad.FwdMode(loss=loss, param=x, seed=[0, 0, 0, 1, 0]):
        ad_fwd_add()

    assert loss.dual[1] == 2


@test_utils.test()
def test_ad_fwd_multiply():
    N = 5
    x = ti.field(ti.f32, shape=N)
    loss = ti.field(ti.f32, shape=N)
    ti.root.lazy_dual()

    for i in range(N):
        x[i] = i

    @ti.kernel
    def ad_fwd_multiply():
        loss[1] += x[3] * x[4]

    with ti.ad.FwdMode(loss=loss, param=x, seed=[0, 0, 0, 1, 1]):
        ad_fwd_multiply()

    assert loss.dual[1] == 7


@test_utils.test()
def test_multiple_calls():
    N = 5
    a = ti.field(float, shape=N)
    b = ti.field(float, shape=N)
    loss_1 = ti.field(float, shape=())
    loss_2 = ti.field(float, shape=())
    ti.root.lazy_dual()

    for i in range(N):
        a[i] = i
        b[i] = i

    @ti.kernel
    def multiple_calls():
        loss_1[None] += 3 * b[1]**2 + 5 * a[3]**2
        loss_2[None] += 4 * b[2]**2 + 6 * a[4]**2

    with ti.ad.FwdMode(loss=loss_1, param=a, seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_1.dual[None] == 30

    with ti.ad.FwdMode(loss=loss_1, param=b, seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_1.dual[None] == 6

    with ti.ad.FwdMode(loss=loss_2, param=b, seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_2.dual[None] == 16

    with ti.ad.FwdMode(loss=loss_2, param=a, seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_2.dual[None] == 48


@test_utils.test()
def test_handle_shape_accessed_by_zero():
    a = ti.field(float)
    b = ti.field(float)
    ti.root.dense(ti.i, 1).place(a, b, a.dual, b.dual)

    @ti.kernel
    def func():
        pass

    with ti.ad.FwdMode(loss=b, param=a):
        func()


@test_utils.test()
def test_handle_shape_accessed_by_none():
    c = ti.field(float, shape=())
    d = ti.field(float, shape=())
    ti.root.lazy_dual()

    @ti.kernel
    def func():
        pass

    with ti.ad.FwdMode(loss=d, param=c):
        func()


@test_utils.test()
def test_clear_all_dual_field():
    x = ti.field(float, shape=(), needs_dual=True)
    y = ti.field(float, shape=(), needs_dual=True)
    loss = ti.field(float, shape=(), needs_dual=True)

    x[None] = 2.0
    y[None] = 3.0

    @ti.kernel
    def clear_dual_test():
        y[None] = x[None]**2
        loss[None] += y[None]

    for _ in range(5):
        with ti.ad.FwdMode(loss=loss, param=x):
            clear_dual_test()
        assert y.dual[None] == 4.0
