import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(exclude=[ti.amdgpu])
def test_abs():
    x = ti.field(ti.f32)

    N = 16

    ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            x[i] = abs(-i)
            print(x[i])
            ti.static_print(x[i])

    func()

    for i in range(N):
        assert x[i] == i


@test_utils.test()
def test_int():
    x = ti.field(ti.f32)

    N = 16

    ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            x[i] = int(x[i])
            x[i] = float(int(x[i]) // 2)

    for i in range(N):
        x[i] = i + 0.4

    func()

    for i in range(N):
        assert x[i] == i // 2
