import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(debug=True)
def test_assign_basic():
    @ti.kernel
    def func_basic():
        a = 1
        assert a == 1

    func_basic()


@test_utils.test(debug=True)
def test_assign_unpack():
    @ti.kernel
    def func_unpack():
        (a, b) = (1, 2)
        assert a == 1
        assert b == 2

    func_unpack()


@test_utils.test(debug=True)
def test_assign_chained():
    @ti.kernel
    def func_chained():
        a = b = 1
        assert a == 1
        assert b == 1

    func_chained()


@test_utils.test(debug=True)
def test_assign_chained_unpack():
    @ti.kernel
    def func_chained_unpack():
        (a, b) = (c, d) = (1, 2)
        assert a == 1
        assert b == 2
        assert c == 1
        assert d == 2

    func_chained_unpack()


@test_utils.test(debug=True)
def test_assign_assign():
    @ti.kernel
    def func_assign():
        a = 0
        a = 1
        assert a == 1

    func_assign()


@test_utils.test(debug=True)
def test_assign_ann():
    @ti.kernel
    def func_ann():
        a: ti.i32 = 1
        b: ti.f32 = a
        assert a == 1
        assert b == 1.0

    func_ann()


@test_utils.test()
def test_assign_ann_over():
    @ti.kernel
    def func_ann_over():
        my_int = ti.i32
        d: my_int = 2
        d: ti.f32 = 2.0

    with pytest.raises(ti.TaichiCompilationError):
        func_ann_over()
