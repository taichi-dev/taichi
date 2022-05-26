import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_log():
    N = 16
    x = ti.field(ti.i32, shape=(N, ))
    y = ti.field(ti.f32, shape=(N, ))

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
    x = ti.field(ti.i32, shape=(N, ))
    y = ti.field(ti.f32, shape=(N, ))

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
    x = ti.field(ti.i32, shape=(N, ))
    y = ti.field(ti.f32, shape=(N, ))

    @ti.kernel
    def func():
        for i in range(N):
            u = ti.sqrt(x[i])
            y[i] = u

    x.from_numpy(np.arange(1, N + 1, dtype=np.int32))

    func()

    assert np.allclose(y.to_numpy(), np.sqrt(x.to_numpy()))


@test_utils.test()
def test_atan2():
    N = 1
    x = ti.field(ti.i32, shape=(N, ))
    y = ti.field(ti.i32, shape=(N, ))

    @ti.kernel
    def test_case_0() -> ti.f32:
        i = ti.i32(2)
        return ti.atan2(i, 1)

    @ti.kernel
    def test_case_1() -> ti.f32:
        x[0] = ti.i32(2)
        return ti.atan2(x[0], 1)

    @ti.kernel
    def test_case_2() -> ti.f32:
        x[0] = ti.i32(3)
        y[0] = ti.i32(1)
        return ti.atan2(x[0], y[0])

    ti_res0 = test_case_0()
    np_res0 = np.arctan2(2, 1)

    ti_res1 = test_case_1()
    np_res1 = np.arctan2(2, 1)

    ti_res2 = test_case_2()
    np_res2 = np.arctan2(3, 1)

    assert np.allclose(ti_res0, np_res0)
    assert np.allclose(ti_res1, np_res1)
    assert np.allclose(ti_res2, np_res2)
