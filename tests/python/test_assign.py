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
        a: ti.i32 = 1
        b: ti.f32 = a
        c = ti.f32
        d: c = 1
        assert a == 1
        assert b == 1.0
        assert d == 1.0

    func_ann()


test_assign_ann()


@ti.test(debug=True)
def test_assign_ann_over():
    @ti.kernel
    def func_ann_over():
        a: ti.i32 = 2
        a: ti.f32 = 2.0
        assert a == 2

    func_ann_over()


test_assign_ann_over()

# @ti.test()
# def test_assign_ann():

#     @ti.kernel
#     def func_ann():
#         a: ti.i32 = 1

#     func_ann()
# try:
#     @ti.kernel
#     def func_ann_overload():
#         a: ti.i32 = 1
#         a : ti.f32 = 2.0
# except ti.SyntaxError:
#     pass
# func_ann_overload()

# @ti.test(debug = True, arch = ti.cpu)
# def test_ann_assign_global():
#     x = ti.field(ti.i32)
#     ti.root.place(x)
#     try:
#         @ti.kernel
#         def func_ann_global():
#             x1: ti.f32 = 1

#     except ti.SyntaxError:
#         pass

#     func_ann_global()
# test_assign_ann()
# test_ann_assign()
# test_ann_assign_global()
