import taichi as ti
import numpy as np
import operator as ops
from taichi import allclose
import pytest

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
        (ti.logical_not, np.logical_not),
        (ti.abs, np.abs),
        (ti.exp, np.exp),
        (ti.log, np.log),
        (ti.sin, np.sin),
        (ti.cos, np.cos),
        (ti.tan, np.tan),
        (ti.asin, np.arcsin),
        (ti.acos, np.arccos),
        (ti.tanh, np.tanh),
        (ti.floor, np.floor),
        (ti.ceil, np.ceil),
]

@pytest.mark.parametrize('ti_func,np_func', binary_func_table)
def test_binary(ti_func, np_func):
    x = ti.Vector([2, 3])
    y = ti.Vector([5, 4])

    result = ti_func(x, y).to_numpy()
    if ti_func in [ops.eq, ops.ne, ops.lt, ops.le, ops.gt, ops.ge]:
        result = result.astype(np.bool)
    expected = np_func(x.to_numpy(), y.to_numpy())
    assert allclose(result, expected)

@pytest.mark.parametrize('ti_func,np_func', unary_func_table)
def test_unary(ti_func, np_func):
    x = ti.Vector([2, 3] if ti_func in [ops.invert, ti.logical_not] else [0.2, 0.3])

    result = ti_func(x).to_numpy()
    if ti_func in [ti.logical_not]:
        result = result.astype(np.bool)
    expected = np_func(x.to_numpy())
    assert allclose(result, expected)
