import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_ad_ndarray_autodiff():
    x = ti.ndarray(ti.f32, shape=(), needs_grad=True)
    y = ti.ndarray(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def write(x: ti.types.ndarray()):
        x[None] = 3.0

    write(x)

    @ti.kernel
    def compute(x: ti.types.ndarray(), y: ti.types.ndarray()):
        y[None] = ti.cos(x[None])

    with ti.ad.TapeNdarray(loss=y):
        compute(x, y)

    @ti.kernel
    def test(x: ti.types.ndarray()):
        assert x.grad[None] == -ti.sin(x[None])

    test(x)
