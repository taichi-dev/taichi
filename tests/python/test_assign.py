import pytest

import taichi as ti


@ti.test(debug=True)
def test_assign_basic():

    @ti.kernel
    def func_basic():
        a = 1
        assert a == 1

    func_basic()


@ti.test(debug=True)
def test_assign_unpack():

    @ti.kernel
    def func_unpack():
        (a, b) = (1, 2)
        assert a == 1
        assert b == 2

    func_unpack()


@ti.test(debug=True)
def test_assign_chained():

    @ti.kernel
    def func_chained():
        a = b = 1
        assert a == 1
        assert b == 1

    func_chained()


@ti.test(debug=True)
def test_assign_chained_unpack():

    @ti.kernel
    def func_chained_unpack():
        (a, b) = (c, d) = (1, 2)
        assert a == 1
        assert b == 2
        assert c == 1
        assert d == 2

    func_chained_unpack()


@ti.test(debug=True)
def test_assign_assign():

    @ti.kernel
    def func_assign():
        a = 0
        a = 1
        assert a == 1

    func_assign()


@ti.test(debug=True)
def test_assign_ann():

    @ti.kernel
    def func_ann():
        # need to introduce ti as a global var
        my_float = ti.f32
        a: ti.i32 = 1
        b: ti.f32 = a
        d: my_float = 1
        assert a == 1
        assert b == 1.0
        assert d == 1.0

    func_ann()


@ti.test()
def test_assign_ann_over():

    @ti.kernel
    def func_ann_over():
        my_int = ti.i32
        d: my_int = 2
        d: ti.f32 = 2.0

    with pytest.raises(ti.TaichiCompilationError):
        func_ann_over()
