import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.gpu])
def test_ad_fwd_add():
    N = 5
    x = ti.field(ti.f32, shape=N)
    loss = ti.field(ti.f32, shape=N)

    for i in range(N):
        x[i] = i

    @ti.kernel
    def ad_fwd_add():
        loss[1] += 2 * x[3]

    with ti.ad.FwdMode(loss=loss, parameters=x, seed=[0, 0, 0, 1, 0]):
        ad_fwd_add()

    assert loss.grad[1] == 2


@test_utils.test(arch=[ti.cpu, ti.gpu])
def test_ad_fwd_multiply():
    N = 5
    x = ti.field(ti.f32, shape=N)
    loss = ti.field(ti.f32, shape=N)

    for i in range(N):
        x[i] = i

    @ti.kernel
    def ad_fwd_multiply():
        loss[1] += x[3] * x[4]

    with ti.ad.FwdMode(loss=loss, parameters=x, seed=[0, 0, 0, 1, 1]):
        ad_fwd_multiply()

    assert loss.grad[1] == 7


@test_utils.test(arch=[ti.cpu, ti.gpu])
def test_multiple_calls():
    N = 5
    a = ti.field(float, shape=N)
    b = ti.field(float, shape=N)
    loss_1 = ti.field(float, shape=())
    loss_2 = ti.field(float, shape=())

    for i in range(N):
        a[i] = i
        b[i] = i

    @ti.kernel
    def multiple_calls():
        loss_1[None] += 3 * b[1]**2 + 5 * a[3]**2
        loss_2[None] += 4 * b[2]**2 + 6 * a[4]**2

    with ti.ad.FwdMode(loss=loss_1, parameters=a,
                       seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_1.grad[None] == 30
    assert not loss_2.snode.ptr.has_dual() and not b.snode.ptr.has_dual()

    with ti.ad.FwdMode(loss=loss_1, parameters=b,
                       seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_1.grad[None] == 6
    assert not loss_2.snode.ptr.has_dual(
    ) and not a.snode.ptr.is_dual_activated()

    with ti.ad.FwdMode(loss=loss_2, parameters=b,
                       seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_2.grad[None] == 16
    assert not loss_1.snode.ptr.is_dual_activated(
    ) and not a.snode.ptr.is_dual_activated()

    with ti.ad.FwdMode(loss=loss_2, parameters=a,
                       seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_2.grad[None] == 48
    assert not loss_1.snode.ptr.is_dual_activated(
    ) and not b.snode.ptr.is_dual_activated()
