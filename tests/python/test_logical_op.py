import taichi as ti
from tests import test_utils


@test_utils.test(debug=True)
def test_logical_and_i32():
    @ti.kernel
    def func(x: ti.i32, y: ti.i32) -> ti.i32:
        return x and y

    assert func(1, 2) == 2
    assert func(2, 1) == 1
    assert func(0, 1) == 0
    assert func(1, 0) == 0


@test_utils.test(debug=True)
def test_logical_or_i32():
    @ti.kernel
    def func(x: ti.i32, y: ti.i32) -> ti.i32:
        return x or y

    assert func(1, 2) == 1
    assert func(2, 1) == 2
    assert func(1, 0) == 1
    assert func(0, 1) == 1


@test_utils.test(debug=True)
def test_logical_and_f32():
    @ti.kernel
    def func(x: ti.f32, y: ti.f32) -> ti.f32:
        return x and y

    assert func(1.5, 2.5) == 2.5
    assert func(2.5, 1.5) == 1.5
    assert func(0, 1.5) == 0
    assert func(1.5, 0) == 0


@test_utils.test(debug=True)
def test_logical_or_f32():
    @ti.kernel
    def func(x: ti.f32, y: ti.f32) -> ti.f32:
        return x or y

    assert func(1.5, 2.5) == 1.5
    assert func(2.5, 1.5) == 2.5
    assert func(1.5, 0) == 1.5
    assert func(0, 1.5) == 1.5
