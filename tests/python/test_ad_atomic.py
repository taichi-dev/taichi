import taichi as ti
from taichi import approx


@ti.test()
def test_ad_reduce():
    N = 16

    x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def func():
        for i in x:
            loss[None] += x[i]**2

    total_loss = 0
    for i in range(N):
        x[i] = i
        total_loss += i * i

    loss.grad[None] = 1
    func()
    func.grad()

    assert total_loss == approx(loss[None])
    for i in range(N):
        assert x.grad[i] == approx(i * 2)
