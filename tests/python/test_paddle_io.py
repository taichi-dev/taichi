import numpy as np
import pytest
from taichi.lang import impl
from taichi.lang.util import has_paddle

import taichi as ti
from tests import test_utils

if has_paddle():
    import paddle


@pytest.mark.skipif(not has_paddle(), reason='Paddle not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_io_devices():
    n = 32
    x = ti.field(dtype=ti.i32, shape=n)

    @ti.kernel
    def load(y: ti.types.ndarray()):
        for i in x:
            x[i] = y[i] + 10

    @ti.kernel
    def inc():
        for i in x:
            x[i] += i

    @ti.kernel
    def store(y: ti.types.ndarray()):
        for i in x:
            y[i] = x[i] * 2

    devices = [paddle.CPUPlace()]
    if paddle.device.is_compiled_with_cuda():
        devices.append(paddle.CUDAPlace(0))
    for device in devices:
        y = paddle.to_tensor(np.ones(shape=n, dtype=np.int32), place=device)

        load(y)
        inc()
        store(y)

        y = y.cpu().numpy()

        for i in range(n):
            assert y[i] == (11 + i) * 2


@pytest.mark.skipif(not has_paddle(), reason='Paddle not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_io():
    n = 32

    @ti.kernel
    def paddle_kernel(zero: ti.types.ndarray()):
        for i in range(n):
            zero[i] += i * i

    x_zero = paddle.zeros(shape=[n], dtype=paddle.int32)
    paddle_kernel(x_zero)
    for i in range(n):
        assert x_zero[i] == i * i


@pytest.mark.skipif(not has_paddle(), reason='Paddle not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_io_2d():
    n = 32

    @ti.kernel
    def paddle_kernel(zero: ti.types.ndarray()):
        for i in range(n):
            for j in range(n):
                zero[i, j] += i * j

    x_zero = paddle.zeros(shape=(n, n), dtype=paddle.int32)
    paddle_kernel(x_zero)
    for i in range(n):
        for j in range(n):
            assert x_zero[i, j] == i * j


@pytest.mark.skipif(not has_paddle(), reason='Paddle not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_io_3d():
    n = 32

    @ti.kernel
    def paddle_kernel(zero: ti.types.ndarray()):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    zero[i, j, k] += i * j * k

    x_zero = paddle.zeros(shape=(n, n, n), dtype=paddle.int32)
    paddle_kernel(x_zero)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert x_zero[i, j, k] == i * j * k


@pytest.mark.skipif(not has_paddle(), reason='Paddle not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_io_simple():
    n = 32

    x1 = ti.field(ti.f32, shape=(n, n))
    p1 = paddle.Tensor(3 * np.ones((n, n), dtype=np.float32))

    x2 = ti.Matrix.field(2, 3, ti.f32, shape=(n, n))
    p2 = paddle.Tensor(3 * np.ones((n, n, 2, 3), dtype=np.float32))

    x1.from_paddle(p1)
    for i in range(n):
        for j in range(n):
            assert x1[i, j] == 3

    x2.from_paddle(p2)
    for i in range(n):
        for j in range(n):
            for k in range(2):
                for l in range(3):
                    assert x2[i, j][k, l] == 3

    p3 = x2.to_paddle()
    assert (p2 == p3).all()


@pytest.mark.skipif(not has_paddle(), reason='Paddle not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_io_zeros():
    mat = ti.Matrix.field(2, 6, dtype=ti.f32, shape=(), needs_grad=True)
    zeros = paddle.zeros((2, 6))
    zeros[1, 2] = 3
    mat.from_paddle(zeros + 1)

    assert mat[None][1, 2] == 4

    zeros = mat.to_paddle()
    assert zeros[1, 2] == 4
