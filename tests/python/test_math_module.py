import errno

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


@ti.func
def check_epsilon_equal(mat_cal, mat_ref, epsilon) -> int:
    assert mat_cal.n == mat_ref.n and mat_cal.m == mat_ref.m
    err = 0
    for i in ti.static(range(mat_cal.n)):
        for j in ti.static(range(mat_cal.m)):
            err = ti.abs(mat_cal[i, j] - mat_ref[i, j]) > epsilon
    return err


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
        ray = Ray(ti.math.vec3(pi), ti.math.vec2(0.5, 0.5), ti.math.mat3(1))

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
        id: ti.math.uvec2

    @ti.kernel
    def test():
        pi = 3.14159265358
        N = ti.u64(2**63 - 1)
        ray = Ray(ti.math.vec3(pi), ti.math.vec2(pi), id=ti.math.uvec2(N))

        assert abs(ray.pos.x - pi) < 1e-10
        assert ray.id.x == N

    test()


@test_utils.test()
@ti.kernel
def test_translate():
    error = 0
    translate_vec = ti.math.vec3(1., 2., 3.)
    translate_mat = ti.math.translate(translate_vec[0], translate_vec[1],
                                      translate_vec[2])
    translate_ref = ti.math.mat4([[1., 0., 0., 1.], [0., 1., 0., 2.],
                                  [0., 0., 1., 3.], [0., 0., 0., 1.]])
    error += check_epsilon_equal(translate_mat, translate_ref, 0.00001)
    assert error == 0


@test_utils.test()
@ti.kernel
def test_scale():
    error = 0
    scale_vec = ti.math.vec3(1., 2., 3.)
    scale_mat = ti.math.scale(scale_vec[0], scale_vec[1], scale_vec[2])
    scale_ref = ti.math.mat4([[1., 0., 0., 0.], [0., 2., 0., 0.],
                              [0., 0., 3., 0.], [0., 0., 0., 1.]])
    error += check_epsilon_equal(scale_mat, scale_ref, 0.00001)
    assert error == 0


@test_utils.test()
@ti.kernel
def test_rotation2d():
    error = 0
    rotationTest = ti.math.rotation2d(ti.math.radians(30))
    rotationRef = ti.math.mat2([[0.866025, -0.500000], [0.500000, 0.866025]])
    error += check_epsilon_equal(rotationRef, rotationTest, 0.00001)
    assert error == 0


@test_utils.test()
@ti.kernel
def test_rotation3d():
    error = 0

    first = 1.046
    second = 0.52
    third = -0.785
    axisX = ti.math.vec3(1.0, 0.0, 0.0)
    axisY = ti.math.vec3(0.0, 1.0, 0.0)
    axisZ = ti.math.vec3(0.0, 0.0, 1.0)

    rotationEuler = ti.math.rot_yaw_pitch_roll(first, second, third)
    rotationInvertedY = ti.math.rot_by_axis(
        axisZ, third) @ ti.math.rot_by_axis(
            axisX, second) @ ti.math.rot_by_axis(axisY, -first)
    rotationDumb = ti.Matrix.zero(ti.f32, 4, 4)
    rotationDumb = ti.math.rot_by_axis(axisY, first) @ rotationDumb
    rotationDumb = ti.math.rot_by_axis(axisX, second) @ rotationDumb
    rotationDumb = ti.math.rot_by_axis(axisZ, third) @ rotationDumb
    rotationTest = ti.math.rotation3d(second, third, first)

    dif0 = rotationEuler - rotationDumb
    dif1 = rotationEuler - rotationInvertedY

    difRef0 = ti.math.mat4([[0.05048351, -0.61339645, -0.78816002, 0.],
                            [0.65833154, 0.61388511, -0.4355969, 0.],
                            [0.75103329, -0.49688014, 0.4348093, 0.],
                            [0., 0., 0., 1.]])
    difRef1 = ti.math.mat4([[-0.60788802, 0., -1.22438441, 0.],
                            [0.60837229, 0., -1.22340979, 0.],
                            [1.50206658, 0., 0., 0.], [0., 0., 0., 0.]])

    error += check_epsilon_equal(dif0, difRef0, 0.00001)
    error += check_epsilon_equal(dif1, difRef1, 0.00001)
    error += check_epsilon_equal(rotationEuler, rotationTest, 0.00001)

    assert error == 0
