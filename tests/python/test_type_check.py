import pytest

import taichi as ti


@ti.test(arch=ti.cpu)
def test_unary_op():
    @ti.kernel
    def floor():
        a = 1
        b = ti.floor(a)

    with pytest.raises(ti.TaichiTypeError,
                       match="'floor' takes real inputs only"):
        floor()


@ti.test(arch=ti.cpu)
def test_binary_op():
    @ti.kernel
    def bitwise_float():
        a = 1
        b = 3.1
        c = a & b

    with pytest.raises(ti.TaichiTypeError,
                       match=r"unsupported operand type\(s\) for '&'"):
        bitwise_float()


# @ti.test(arch=ti.cpu)
# def test_ternary_op():
#     @ti.kernel
#     def select():
#         a = 1.1
#         b = 3
#         c = 3.6
#         d = b if a else c
#
#     with pytest.raises(ti.TaichiCompilationError,
#                        match="for 'select': 'f32', 'i32' and 'f32'"):
#         select()


@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=[ti.cpu, ti.opengl])
def test_subscript():
    a = ti.ndarray(ti.i32, shape=(10, 10))

    @ti.kernel
    def any_array(x: ti.any_arr()):
        b = x[3, 1.1]

    with pytest.raises(ti.TaichiTypeError, match="indices must be integers"):
        any_array(a)
