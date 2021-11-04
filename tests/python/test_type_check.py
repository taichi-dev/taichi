import pytest

import taichi as ti


@ti.test(arch=ti.cpu)
def test_unary_op():
    @ti.kernel
    def floor():
        a = 1
        b = ti.floor(a)

    with pytest.raises(SystemExit):
        floor()


@ti.test(arch=ti.cpu)
def test_binary_op():
    @ti.kernel
    def bitwise_float():
        a = 1
        b = 3.1
        c = a & b

    with pytest.raises(SystemExit):
        bitwise_float()


@ti.test(arch=ti.cpu)
def test_ternary_op():
    @ti.kernel
    def select():
        a = 1.1
        b = 3
        c = 3.6
        d = b if a else c

    with pytest.raises(SystemExit):
        select()
