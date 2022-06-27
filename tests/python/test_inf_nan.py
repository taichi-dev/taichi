import taichi as ti
from taichi.math import nan, inf, isnan, isinf
from tests import test_utils


def _test_inf_nan(dt):
    ti.init(default_fp=dt)

    @ti.kernel
    def make_tests():
        assert isnan(nan) == isnan(-nan) == True
        x = -1.0
        assert isnan(ti.sqrt(x)) == True
        assert isnan(inf) == isnan(1.0) == isnan(-1) == False
        assert isinf(inf) == isinf(-inf) == True
        assert isinf(nan) == isinf(1.0) == isinf(-1) == False

    make_tests()


@test_utils.test(default_fp=ti.f32)
def test_inf_nan_f32():
    _test_inf_nan(ti.f32)


@test_utils.test(default_fp=ti.f64)
def test_inf_nan_f64():
    _test_inf_nan(ti.f64)