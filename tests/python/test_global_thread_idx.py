import taichi as ti
import numpy as np


@ti.test(arch=[ti.cuda, ti.cpu])
def test_thread_idx():
    x = ti.field(ti.i32, shape=(256))

    @ti.kernel
    def func():
        for i in range(32):
            for j in range(8):
                t = ti.global_thread_idx()
                x[t] += 1

    func()
    assert x.to_numpy().sum() == 256

@ti.test(arch=ti.cuda)
def test_global_thread_idx():
    n = 2048
    x = ti.field(ti.i32, shape=(n))

    @ti.kernel
    def func():
        for i in range(n):
            tid = ti.global_thread_idx()
            x[tid] = tid
    
    func()
    assert np.arange(n).sum() == x.to_numpy().sum()
