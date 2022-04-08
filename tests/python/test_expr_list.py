import taichi as ti
from tests import test_utils


@test_utils.test(ti.cpu)
def test_expr_list_basic():
    @ti.kernel
    def func(u: int, v: float) -> float:
        x = [2 + u, 3 + v]
        return x[0] * 100 + x[1]

    assert func(1, 1.1) == test_utils.approx(304.1)


@test_utils.test()
def test_listcomp_multiple_ifs():
    x = ti.field(ti.i32, shape=(4, ))

    @ti.kernel
    def test() -> ti.i32:
        # Taichi doesn't support global fields appearing anywhere after "for"
        # here.
        a = [x[0] for j in range(100) if j > 2 if j < 5]
        return sum(a)

    for i in range(6):
        x[0] = i
        assert test() == i * 2
