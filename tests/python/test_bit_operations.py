import taichi as ti
import numpy as np
import operator as ops
from taichi import allclose
import pytest


@ti.test()
def test_bit_shl():
    @ti.kernel
    def shl(a: ti.i32, b: ti.i32) -> ti.i32:
        return a << b

    for i in range(8):
        assert shl(3, i) == 3 * 2**i


@ti.test()
def test_bit_sar():
    @ti.kernel
    def sar(a: ti.i32, b: ti.i32) -> ti.i32:
        return a >> b

    for i in range(8):
        assert sar(2**8, i) == 2 ** (8 - i)