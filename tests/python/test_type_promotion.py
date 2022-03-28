import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_log():
    N = 16
    x = ti.field(ti.i32, shape=(N,))
    y = ti.field(ti.f32, shape=(N,))

    @ti.kernel
    def func():
        for i in range(N):
            u = ti.log(x[i])
            y[i] = u

    x.from_numpy(np.arange(1, N + 1, dtype=np.int32))

    func()

    assert np.allclose(y.to_numpy(), np.log(x.to_numpy()))


@test_utils.test()
def test_exp():
    N = 16
    x = ti.field(ti.i32, shape=(N,))
    y = ti.field(ti.f32, shape=(N,))

    @ti.kernel
    def func():
        for i in range(N):
            u = ti.exp(x[i])
            y[i] = u

    x.from_numpy(np.arange(1, N + 1, dtype=np.int32))

    func()

    assert np.allclose(y.to_numpy(), np.exp(x.to_numpy()))

@test_utils.test()
def test_sqrt():
    N = 16
    x = ti.field(ti.i32, shape=(N,))
    y = ti.field(ti.f32, shape=(N,))

    @ti.kernel
    def func():
        for i in range(N):
            u = ti.sqrt(x[i])
            y[i] = u

    x.from_numpy(np.arange(1, N + 1, dtype=np.int32))

    func()

    assert np.allclose(y.to_numpy(), np.sqrt(x.to_numpy()))
