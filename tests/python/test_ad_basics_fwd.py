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

    with ti.fwdAD(loss=[loss], parameters=x, seed=[0, 0, 0, 1, 0]):
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

    with ti.fwdAD(loss=[loss], parameters=x, seed=[0, 0, 0, 1, 1]):
        ad_fwd_multiply()

    assert loss.grad[1] == 7
