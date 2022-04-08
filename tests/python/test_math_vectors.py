import taichi as ti
from tests import test_utils


@test_utils.test()
def test_vector_swizzle_python():
    a = ti.math.vec3(1, 2, 3)
    assert all(a.gbr == (2, 3, 1))
    a.bgr = (2, 3, 3)
    assert all(a.rgb == (3, 3, 2))
    b = ti.math.vec3(1)
    assert all((a + b).rgb == (4, 4, 3))
    M = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert all((M @ b) == (1, 1, 1))


@test_utils.test(debug=True)
def test_vector_swizzle_taichi():
    @ti.kernel
    def foo():
        a = ti.math.vec3(1, 2, 3)
        assert all(a.gbr == (2, 3, 1))
        a.bgr = (2, 3, 3)
        assert all(a.rgb == (3, 3, 2))
        b = ti.math.vec3(1)
        assert all((a + b).rgb == (4, 4, 3))
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
