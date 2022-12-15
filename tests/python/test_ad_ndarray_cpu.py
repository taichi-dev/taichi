import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_ad_ndarray_cpu():
    x = ti.ndarray(dtype=ti.f32, shape=(4), needs_grad=True)

    @ti.kernel
    def compute_ndarray(x: ti.types.ndarray()):
        x.grad[0] = float(x.grad.shape[0])
        x[0] = 70.0

    compute_ndarray(x)

    assert x[0] == 70.0
    assert x.grad[0] == 4
