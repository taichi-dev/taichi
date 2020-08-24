import taichi as ti
import numpy as np
import operator as ops
from taichi import allclose
import pytest

func_table = [
    (ops.lshift, ) * 2,
]


@pytest.mark.parametrize('ti_func,np_func', func_table)
def test_python_scope_vector_binary_bit_operations(ti_func, np_func):
    x = ti.Vector([2, 3])
    y = ti.Vector([5, 4])

    result = ti_func(x, y).to_numpy()
    expected = np_func(x.to_numpy(), y.to_numpy())
    assert allclose(result, expected)
