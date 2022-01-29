import numpy as np
import pytest
from taichi.lang.util import has_pytorch

import taichi as ti


@ti.test(arch=ti.cpu)
def test_unary_op():
    @ti.kernel
    def floor():
        a = 1
        b = ti.floor(a)

    with pytest.raises(
            ti.TaichiTypeError,
            match="`@operandType` needs to be Real, but i32 is not"):
        floor()


@ti.test(arch=ti.cpu)
def test_binary_op():
    @ti.kernel
    def bitwise_float():
        a = 1
        b = 3.1
        c = a & b

    with pytest.raises(ti.TaichiTypeError,
                       match=r"`@RHS` needs to be Integral, but f32 is not"):
        bitwise_float()


@ti.test(arch=ti.cpu)
def test_ternary_op():
    @ti.kernel
    def select():
        a = 1.1
        b = 3
        c = 3.6
        d = b if a else c

    with pytest.raises(TypeError,
                       match="`if` conditions must be of type int32"):
        select()


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=[ti.cpu, ti.opengl])
def test_subscript():
    a = ti.ndarray(ti.i32, shape=(10, 10))

    @ti.kernel
    def any_array(x: ti.any_arr()):
        b = x[3, 1.1]

    with pytest.raises(ti.TaichiTypeError, match="indices must be integers"):
        any_array(a)


@ti.test()
def test_0d_ndarray():
    @ti.kernel
    def foo() -> ti.i32:
        a = np.array(3, dtype=np.int32)
        return a

    assert foo() == 3


@ti.test()
def test_non_0d_ndarray():
    @ti.kernel
    def foo():
        a = np.array([1])

    with pytest.raises(
            ti.TaichiTypeError,
            match=
            "Only 0-dimensional numpy array can be used to initialize a scalar expression"
    ):
        foo()
