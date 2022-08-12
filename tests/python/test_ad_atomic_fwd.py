import taichi as ti
from tests import test_utils


@test_utils.test()
def test_ad_reduce_fwd():
    N = 16

    x = ti.field(dtype=ti.f32, shape=N)
    loss = ti.field(dtype=ti.f32, shape=())
    ti.root.lazy_dual()

    @ti.kernel
    def func():
        for i in x:
            loss[None] += x[i]**2

    total_loss = 0
    for i in range(N):
        x[i] = i
        total_loss += i * i

    with ti.ad.FwdMode(loss=loss, param=x, seed=[1.0 for _ in range(N)]):
        func()

    assert total_loss == test_utils.approx(loss[None])
    sum = 0
    for i in range(N):
        sum += i * 2

    assert loss.dual[None] == test_utils.approx(sum)
