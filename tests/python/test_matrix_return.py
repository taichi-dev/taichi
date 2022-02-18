import os

import pytest

import taichi as ti
from tests import test_utils

### `ti.test`


@test_utils.test()
def test_arch():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i32):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert func()[1, 2] == 6


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.metal])
def test_ret_i16():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i16):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert func()[1, 2] == 6


@test_utils.test()
def test_arch_exceed_limit():
    @ti.kernel
    def func() -> ti.types.matrix(3, 10, ti.i32):
        return ti.Matrix([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                          [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])

    assert func()[1, 2] == 12
