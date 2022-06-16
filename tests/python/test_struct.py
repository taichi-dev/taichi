import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize("round", range(10))
@test_utils.test()
def test_linear(round):
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.dense(ti.i, n).place(y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@test_utils.test()
def test_linear_nested():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(x)
    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@test_utils.test()
def test_linear_nested_aos():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(x, y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@test_utils.test(exclude=[ti.vulkan, ti.dx11])
def test_2d_nested():
    x = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.ij, n // 16).dense(ti.ij, (32, 16)).place(x)

    for i in range(n * 2):
        for j in range(n):
            x[i, j] = i + j * 10

    for i in range(n * 2):
        for j in range(n):
            assert x[i, j] == i + j * 10
