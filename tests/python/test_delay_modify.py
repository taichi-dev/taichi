import taichi as ti
from tests import test_utils


@test_utils.test()
def test_simplify_bug():
    @ti.kernel
    def foo() -> ti.types.vector(4, dtype=ti.i32):
        a = ti.Vector([0, 0, 0, 0])
        for i in range(5):
            for k in ti.static(range(4)):
                if i == 3:
                    a[k] = 1
        return a

    a = foo()

    assert (a == ti.Vector([1, 1, 1, 1])).all() == 1
