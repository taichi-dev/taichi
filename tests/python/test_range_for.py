import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_range_for():
    x = ti.field(ti.f32, shape=(16))

    @ti.kernel
    def func():
        for i in range(4, 10):
            x[i] = i
        for i in range(15, 0):
            if 4 <= i < 10:
                assert x[i] == i
            else:
                assert x[i] == 0

    func()
