import numpy as np

import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda)
def test_global_thread_idx():
    n = 2048
    x = ti.field(ti.i32, shape=n)

    @ti.kernel
    def func():
        for i in range(n):
            tid = ti.global_thread_idx()
            x[tid] = tid

    func()
    assert np.arange(n).sum() == x.to_numpy().sum()
