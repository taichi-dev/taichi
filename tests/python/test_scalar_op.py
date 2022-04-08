import operator as ops

import numpy as np
import pytest

import taichi as ti
from tests import test_utils

binary_func_table = [
    (ops.add, ) * 2,
    (ops.sub, ) * 2,
    (ops.mul, ) * 2,
    (ops.truediv, ) * 2,
    (ops.floordiv, ) * 2,
    (ops.mod, ) * 2,
    (ops.pow, ) * 2,
    (ops.and_, ) * 2,
    (ops.or_, ) * 2,
    (ops.xor, ) * 2,
    (ops.eq, ) * 2,
    (ops.ne, ) * 2,
    (ops.lt, ) * 2,
    (ops.le, ) * 2,
    (ops.gt, ) * 2,
    (ops.ge, ) * 2,
    (ti.max, np.maximum),
    (ti.min, np.minimum),
    (ti.atan2, np.arctan2),
]

unary_func_table = [
    (ops.neg, ) * 2,
    (ops.invert, ) * 2,
    (ti.lang.ops.logical_not, np.logical_not),
    (ti.lang.ops.abs, np.abs),
    (ti.exp, np.exp),
    (ti.log, np.log),
    (ti.sin, np.sin),
    (ti.cos, np.cos),
    (ti.tan, np.tan),
    (ti.asin, np.arcsin),
    (ti.acos, np.arccos),
    (ti.tanh, np.tanh),
    (ti.round, np.round),
    (ti.floor, np.floor),
    (ti.ceil, np.ceil),
]


@pytest.mark.parametrize('ti_func,np_func', binary_func_table)
def test_python_scope_vector_binary(ti_func, np_func):
    ti.init()
    x = ti.Vector([2, 3])
    y = ti.Vector([5, 4])

    result = ti_func(x, y).to_numpy()
    if ti_func in [ops.eq, ops.ne, ops.lt, ops.le, ops.gt, ops.ge]:
        result = result.astype(bool)
    expected = np_func(x.to_numpy(), y.to_numpy())
    assert test_utils.allclose(result, expected)


@pytest.mark.parametrize('ti_func,np_func', unary_func_table)
def test_python_scope_vector_unary(ti_func, np_func):
    ti.init()
    x = ti.Vector([2, 3] if ti_func in
                  [ops.invert, ti.lang.ops.logical_not] else [0.2, 0.3])

    result = ti_func(x).to_numpy()
    if ti_func in [ti.lang.ops.logical_not]:
        result = result.astype(bool)
    expected = np_func(x.to_numpy())
    assert test_utils.allclose(result, expected)


def test_python_scope_matmul():
    ti.init()
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    x = ti.Vector(a)
    y = ti.Vector(b)

    result = (x @ y).to_numpy()
    expected = a @ b
    assert test_utils.allclose(result, expected)


def test_python_scope_linalg():
    ti.init()
    a = np.array([3, 4, -2])
    b = np.array([-5, 0, 6])
    x = ti.Vector(a)
    y = ti.Vector(b)

    assert test_utils.allclose(x.dot(y), np.dot(a, b))
    assert test_utils.allclose(x.norm(), np.sqrt(np.dot(a, a)))
    assert test_utils.allclose(x.normalized(), a / np.sqrt(np.dot(a, a)))
    assert x.any() == 1  # To match that of Taichi IR, we return -1 for True
    assert y.all() == 0


@test_utils.test(arch=[ti.x64, ti.cuda, ti.metal])
def test_16_min_max():
    @ti.kernel
    def min_u16(a: ti.u16, b: ti.u16) -> ti.u16:
        return ti.min(a, b)

    @ti.kernel
    def min_i16(a: ti.i16, b: ti.i16) -> ti.i16:
        return ti.min(a, b)

    @ti.kernel
    def max_u16(a: ti.u16, b: ti.u16) -> ti.u16:
        return ti.max(a, b)

    @ti.kernel
    def max_i16(a: ti.i16, b: ti.i16) -> ti.i16:
        return ti.max(a, b)

    a, b = 4, 2
    assert min_u16(a, b) == min(a, b)
    assert min_i16(a, b) == min(a, b)
    assert max_u16(a, b) == max(a, b)
    assert max_i16(a, b) == max(a, b)


@test_utils.test(exclude=[ti.opengl, ti.cc])
def test_32_min_max():
    @ti.kernel
    def min_u32(a: ti.u32, b: ti.u32) -> ti.u32:
        return ti.min(a, b)

    @ti.kernel
    def min_i32(a: ti.i32, b: ti.i32) -> ti.i32:
        return ti.min(a, b)

    @ti.kernel
    def max_u32(a: ti.u32, b: ti.u32) -> ti.u32:
        return ti.max(a, b)

    @ti.kernel
    def max_i32(a: ti.i32, b: ti.i32) -> ti.i32:
        return ti.max(a, b)

    a, b = 4, 2
    assert min_u32(a, b) == min(a, b)
    assert min_i32(a, b) == min(a, b)
    assert max_u32(a, b) == max(a, b)
    assert max_i32(a, b) == max(a, b)


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_64_min_max():
    @ti.kernel
    def min_u64(a: ti.u64, b: ti.u64) -> ti.u64:
        return ti.min(a, b)

    @ti.kernel
    def min_i64(a: ti.i64, b: ti.i64) -> ti.i64:
        return ti.min(a, b)

    @ti.kernel
    def max_u64(a: ti.u64, b: ti.u64) -> ti.u64:
        return ti.max(a, b)

    @ti.kernel
    def max_i64(a: ti.i64, b: ti.i64) -> ti.i64:
        return ti.max(a, b)

    a, b = 4, 2
    assert min_u64(a, b) == min(a, b)
    assert min_i64(a, b) == min(a, b)
    assert max_u64(a, b) == max(a, b)
    assert max_i64(a, b) == max(a, b)


@test_utils.test()
def test_min_max_vector_starred():
    @ti.kernel
    def min_starred() -> ti.i32:
        a = ti.Vector([1, 2, 3])
        b = ti.Vector([4, 5, 6])
        return ti.min(*a, *b)

    @ti.kernel
    def max_starred() -> ti.i32:
        a = ti.Vector([1, 2, 3])
        b = ti.Vector([4, 5, 6])
        return ti.max(*a, *b)

    assert min_starred() == 1
    assert max_starred() == 6
