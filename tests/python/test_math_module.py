import pytest

import taichi as ti
from taichi.math import inf, isinf, isnan, nan, pi, vdir
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


@test_utils.test()
def test_vdir():
    @ti.kernel
    def make_test():
        assert all(vdir(pi / 2) == [0, 1])

    make_test()


@test_utils.test(default_fp=ti.f32, debug=True)
def test_vector_types_f32():

    @ti.dataclass
    class Ray:

        pos: ti.math.vec3
        uv: ti.math.vec2
        mat: ti.math.mat3
        _id: ti.math.uvec2

    @ti.kernel
    def test():
        ray = Ray(ti.math.vec3(pi),
                  ti.math.vec2(0.5, 0.5),
                  ti.math.mat3(1))

    test()


@test_utils.test(require=ti.extension.data64,
                 default_fp=ti.f64,
                 default_ip=ti.i64,
                 debug=True)
def test_vector_types_f64():

    @ti.dataclass
    class Ray:

        pos: ti.math.vec3
        uv: ti.math.vec2
        mat: ti.math.mat3
        _id: ti.math.uvec2

    @ti.kernel
    def test():
        pi = 3.14159265358
        N = ti.u64(2**63 - 1)
        ray = Ray(ti.math.vec3(pi),
                  ti.math.vec2(pi),
                  id=ti.math.uvec2(N))

        assert abs(ray.pos.x - pi) < 1e-10
        assert ray.id.x == N

    test()
