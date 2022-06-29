import taichi as ti
from tests import test_utils


@test_utils.test()
def test_ad_reduce_fwd():
    N = 16

    x = ti.field(dtype=ti.f32, shape=N)
    loss = ti.field(dtype=ti.f32, shape=N)
    ti.root.lazy_dual()

    @ti.kernel
    def func():
        for i in x:
            loss[i] += x[i]**2

    total_loss = 0
    for i in range(N):
        x[i] = i
        total_loss += i * i

    with ti.ad.FwdMode(loss=loss, parameters=x, seed=[1.0 for _ in range(N)]):
        func()

    total_loss_computed = 0
    for i in range(N):
        total_loss_computed += loss[i]

    assert total_loss == test_utils.approx(total_loss_computed)
    for i in range(N):
        assert loss.dual[i] == test_utils.approx(i * 2)
