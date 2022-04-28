import numpy as np
import pytest
from taichi.lang import impl
from taichi.lang.util import has_paddle

import taichi as ti
from tests import test_utils

if has_paddle():
    import paddle


@pytest.mark.skipif(not has_paddle(), reason='PaddlePaddle not installed.')
@test_utils.test(exclude=[ti.opengl, ti.vulkan])
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
