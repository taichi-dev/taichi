import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_ad_ndarray_cpu():
    x = ti.ndarray(dtype=ti.f32, shape=(4), needs_grad=True)

    @ti.kernel
    def compute_ndarray(x: ti.types.ndarray()):
        x.grad[0] = 10.0
        x.grad[1] = 20.0
        x.grad[2] = 30.0
        x.grad[3] = 40.0
        x[0] = 50.0
        x[1] = 60.0
        x[2] = 70.0
        x[3] = 80.0

    compute_ndarray(x)

    assert x.grad[0] == 10.0
    assert x.grad[1] == 20.0
    assert x.grad[2] == 30.0
    assert x.grad[3] == 40.0
    assert x[0] == 50.0
    assert x[1] == 60.0
    assert x[2] == 70.0
    assert x[3] == 80.0
