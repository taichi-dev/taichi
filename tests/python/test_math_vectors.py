import taichi as ti
from tests import test_utils


@test_utils.test()
def test_vector_swizzle_python():
    a = ti.math.vec3(1, 2, 3)
    assert all(a.gbr == (2, 3, 1))
    a.bgr = (2, 3, 3)
    assert all(a.rgb == (3, 3, 2))
    a.gbr = a.gbr + (
        1, 2, 3
    )  # FIXME: Taichi does not support += on matrices in python scope
    assert all(a.rgb == (6, 4, 4))
    b = ti.math.vec3(1)
    assert all((a + b).rrb == (7, 7, 5))
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
        a.gbr += (1, 2, 3)
        assert all(a.rgb == (6, 4, 4))
        b = ti.math.vec3(1)
        assert all((a + b).rrb == (7, 7, 5))
        M = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert all((M @ b) == (1, 1, 1))

    foo()
