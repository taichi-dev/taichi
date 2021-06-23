import operator as ops

import numpy as np
import pytest

import taichi as ti
from taichi import allclose


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

    n = 8
    test_num = 2**n
    neg_test_num = -test_num
    for i in range(n):
        assert sar(test_num, i) == 2**(n - i)
    # for negative number
    for i in range(n):
        assert sar(neg_test_num, i) == -2**(n - i)


@ti.test()
def test_bit_shr():
    @ti.kernel
    def shr(a: ti.i32, b: ti.i32) -> ti.i32:
        return ti.bit_shr(a, b)

    n = 8
    test_num = 2**n
    neg_test_num = -test_num
    for i in range(n):
        assert shr(test_num, i) == 2**(n - i)
    for i in range(n):
        offset = 0x100000000 if i > 0 else 0
        assert shr(neg_test_num, i) == (neg_test_num + offset) >> i
