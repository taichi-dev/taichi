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
