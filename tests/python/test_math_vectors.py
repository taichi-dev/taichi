import pytest

import taichi as ti
from tests import test_utils


def _Vector_based_vec3_maker(*data):
    if len(data) == 1:
        data = data * 3
    return ti.Vector(data, dt=ti.f32)


vec3_makers = [
    ti.math.vec3,
    _Vector_based_vec3_maker,
]


@pytest.mark.parametrize('make_vec3', vec3_makers)
@test_utils.test()
def test_vector_swizzle_python(make_vec3):
    a = make_vec3(1, 2, 3)
    assert all(a.gbr == (2, 3, 1))
    a.bgr = (2, 3, 3)
    assert all(a.rgb == (3, 3, 2))
    a.gbr = a.gbr + (
        1, 2, 3
    )  # FIXME: Taichi does not support += on matrices in python scope
    assert all(a.rgb == (6, 4, 4))
    b = make_vec3(1)
    assert all((a + b).rrb == (7, 7, 5))
    M = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert all((M @ b) == (1, 1, 1))


@pytest.mark.parametrize('make_vec3', vec3_makers)
@test_utils.test(debug=True)
def test_vector_swizzle_taichi(make_vec3):
    @ti.kernel
    def foo():
        a = make_vec3(1, 2, 3)
        assert all(a.gbr == (2, 3, 1))
        a.bgr = (2, 3, 3)
        assert all(a.rgb == (3, 3, 2))
        a.gbr += (1, 2, 3)
        assert all(a.rgb == (6, 4, 4))
        b = make_vec3(1)
        assert all((a + b).rrb == (7, 7, 5))
        M = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert all((M @ b) == (1, 1, 1))

    foo()


@test_utils.test(debug=True)
def test_vector_dtype():
    @ti.kernel
    def foo():
        a = ti.math.vec3(1, 2, 3)
        a /= 2
        assert all(abs(a - (0.5, 1., 1.5)) < 1e-6)
        b = ti.math.ivec3(1.5, 2.5, 3.5)
        assert all(b == (1, 2, 3))

    foo()
