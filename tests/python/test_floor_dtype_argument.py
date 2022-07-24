import taichi as ti
from tests import test_utils


def _test_floor_ceil_round(dt):
    @ti.kernel
    def make_tests():
        x = 1.5
        v = ti.math.vec3(1.1, 2.2, 3.3)

        assert ti.floor(x) == 1
        assert ti.floor(x, dt) == 1.0
        assert ti.floor(x, int) == 1

        assert all(ti.floor(v) == [1, 2, 3])
        assert all(ti.floor(v, dt) == [1.0, 2.0, 3.0])
        assert all(ti.floor(v, int) == [1, 2, 3])

        assert ti.ceil(x) == 2
        assert ti.ceil(x, dt) == 2.0
        assert ti.ceil(x, int) == 2

        assert all(ti.ceil(v) == [2, 3, 4])
        assert all(ti.ceil(v, dt) == [2.0, 3.0, 4.0])
        assert all(ti.ceil(v, int) == [2, 3, 4])

        assert ti.round(x) == 2
        assert ti.round(x, dt) == 2.0
        assert ti.round(x, int) == 2

        assert all(ti.round(v) == [1, 2, 3])
        assert all(ti.round(v, dt) == [1.0, 2.0, 3.0])
        assert all(ti.round(v, int) == [1, 2, 3])

    make_tests()


@test_utils.test(default_fp=ti.f32)
def test_floor_ceil_round_f32():
    _test_floor_ceil_round(ti.f32)


@test_utils.test(default_fp=ti.f64, require=ti.extension.data64)
def test_floor_ceil_round_f64():
    _test_floor_ceil_round(ti.f64)
