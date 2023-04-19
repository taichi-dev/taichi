import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(default_fp=ti.f64, exclude=[ti.vulkan, ti.opengl, ti.gles, ti.metal])
def test_general():
    x1 = ti.field(dtype=float, shape=(2, 2), needs_grad=True)
    y1 = ti.field(dtype=float, shape=(), needs_grad=True)

    x1.from_numpy(np.array([[1, 2], [3, 4]]))

    @ti.kernel
    def compute_y1():
        for i, j in ti.ndrange(2, 2):
            y1[None] += ti.cos(x1[i, j])

    x2 = ti.Vector.field(n=3, dtype=float, shape=(2, 2), needs_grad=True)
    y2 = ti.field(dtype=float, shape=(), needs_grad=True)
    x2[0, 0] = ti.Vector([1, 2, 3])
    x2[0, 1] = ti.Vector([4, 5, 6])
    x2[1, 0] = ti.Vector([7, 8, 9])
    x2[1, 1] = ti.Vector([10, 11, 12])

    @ti.kernel
    def compute_y2():
        y2[None] += x2[0, 0][0] + x2[1, 0][1] + x2[1, 1][2]

    with ti.ad.Tape(y1, grad_check=[x1]):
        compute_y1()

    with ti.ad.Tape(y2, grad_check=[x2]):
        compute_y2()


def grad_test(tifunc):
    print(f"arch={ti.lang.impl.current_cfg().arch} default_fp={ti.lang.impl.current_cfg().default_fp}")
    x = ti.field(ti.lang.impl.current_cfg().default_fp)
    y = ti.field(ti.lang.impl.current_cfg().default_fp)

    ti.root.place(x, x.grad, y, y.grad)

    @ti.kernel
    def func():
        for i in ti.grouped(x):
            y[i] = tifunc(x[i])

    x[None] = 0.234

    with ti.ad.Tape(loss=y, grad_check=[x]):
        func()


@pytest.mark.parametrize(
    "tifunc",
    [
        lambda x: x,
        lambda x: ti.abs(-x),
        lambda x: -x,
        lambda x: x * x,
        lambda x: x**2,
        lambda x: x * x * x,
        lambda x: x * x * x * x,
        lambda x: 0.4 * x * x - 3,
        lambda x: (x - 3) * (x - 1),
        lambda x: (x - 3) * (x - 1) + x * x,
        lambda x: ti.tanh(x),
        lambda x: ti.sin(x),
        lambda x: ti.cos(x),
        lambda x: ti.acos(x),
        lambda x: ti.asin(x),
        lambda x: 1 / x,
        lambda x: (x + 1) / (x - 1),
        lambda x: (x + 1) * (x + 2) / ((x - 1) * (x + 3)),
        lambda x: ti.sqrt(x),
        lambda x: ti.exp(x),
        lambda x: ti.log(x),
        lambda x: ti.min(x, 0),
        lambda x: ti.min(x, 1),
        lambda x: ti.min(0, x),
        lambda x: ti.min(1, x),
        lambda x: ti.max(x, 0),
        lambda x: ti.max(x, 1),
        lambda x: ti.max(0, x),
        lambda x: ti.max(1, x),
        lambda x: ti.atan2(0.4, x),
        lambda x: ti.atan2(x, 0.4),
        lambda x: 0.4**x,
        lambda x: x**0.4,
    ],
)
@test_utils.test(default_fp=ti.f64, exclude=[ti.vulkan, ti.opengl, ti.gles, ti.metal])
def test_basics(tifunc):
    grad_test(tifunc)
