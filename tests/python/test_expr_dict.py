import taichi as ti
from tests import test_utils


@test_utils.test(ti.cpu)
def test_expr_dict_basic():
    @ti.kernel
    def func(u: int, v: float) -> float:
        x = {'foo': 2 + u, 'bar': 3 + v}
        return x['foo'] * 100 + x['bar']

    assert func(2, 0.1) == test_utils.approx(403.1)


@test_utils.test(ti.cpu)
def test_expr_dict_field():
    a = ti.field(ti.f32, shape=(4, ))

    @ti.kernel
    def func() -> float:
        x = {'foo': 2 + a[0], 'bar': 3 + a[1]}
        return x['foo'] * 100 + x['bar']

    a[0] = 2
    a[1] = 0.1
    assert func() == test_utils.approx(403.1)


@test_utils.test(ti.cpu)
def test_dictcomp_multiple_ifs():
    n = 8
    x = ti.field(ti.i32, shape=(n, ))

    @ti.kernel
    def test() -> ti.i32:
        # Taichi doesn't support global fields appearing anywhere after "for"
        # here.
        a = {x[j]: x[j] + j for j in range(100) if j > 2 if j < 5}
        return sum(a.values())

    for i in range(n):
        x[i] = i * 2

    assert test() == (3 * 2 + 3) + (4 * 2 + 4)
