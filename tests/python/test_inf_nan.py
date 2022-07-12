import pytest
import taichi as ti
from taichi.math import inf, isinf, isnan, nan
from tests import test_utils


def _test_inf_nan(dt):
    @ti.kernel
    def make_tests():
        assert isnan(nan) == isnan(-nan) == True
        x = -1.0
        assert isnan(ti.sqrt(x)) == True
        assert isnan(inf) == isnan(1.0) == isnan(-1) == False
        assert isinf(inf) == isinf(-inf) == True
        assert isinf(nan) == isinf(1.0) == isinf(-1) == False

        v = ti.math.vec4(inf, -inf, 1.0, nan)
        assert all(isinf(v) == [1, 1, 0, 0])

        v = ti.math.vec4(nan, -nan, 1, inf)
        assert all(isnan(v) == [1, 1, 0, 0])

    make_tests()


@pytest.mark.parametrize('dt', [ti.f32, ti.f64])
@test_utils.test()
def test_inf_nan_f32(dt):
    _test_inf_nan(dt)