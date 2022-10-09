import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(default_fp=ti.f64, exclude=[ti.cc, ti.vulkan, ti.opengl])
def test_grad_check():
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
