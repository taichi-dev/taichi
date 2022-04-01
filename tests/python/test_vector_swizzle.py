import taichi as ti
from tests import test_utils


@test_utils.test()
def test_vector_swizzle_python():
    a = ti.vec3(1, 2, 3)
    assert all(a.gbr == (2, 3, 1))
    a.bgr = (2, 3, 3)
    assert all(a.rgb == (3, 3, 2))
    a.r = 1
    assert a.r == 1


@test_utils.test(debug=True)
def test_vector_swizzle_taichi():
    @ti.kernel
    def foo():
        a = ti.vec3(1, 2, 3)
        assert all(a.gbr == (2, 3, 1))
        a.bgr = (2, 3, 3)
        assert all(a.rgb == (3, 3, 2))
        a.r = 1
        assert a.r == 1

    foo()
