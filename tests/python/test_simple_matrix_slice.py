import taichi as ti
from tests import test_utils


@test_utils.test()
def test_slice():
    b = 3

    @ti.kernel
    def foo1() -> ti.types.vector(3, dtype=ti.i32):
        c = ti.Vector([0, 1, 2, 3, 4, 5, 6])
        return c[:5:2]

    @ti.kernel
    def foo2() -> ti.types.matrix(2, 2, dtype=ti.i32):
        a = ti.Matrix([[1, 2, 3], [4, 5, 6]])
        return a[:, :b:2]

    v1 = foo1()
    assert v1[0] == 0
    assert v1[2] == 4
    m1 = foo2()
    assert m1[0, 0] == 1
    assert m1[1, 0] == 4
