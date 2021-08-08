import numpy as np

import taichi as ti


@ti.test(require=ti.extension.async_mode, async_mode=True)
def test_simple():
    n = 32

    x = ti.field(dtype=ti.i32, shape=n)

    @ti.kernel
    def double():
        for i in x:
            x[i] = i * 2

    double()

    for i in range(n):
        assert x[i] == i * 2


@ti.test(require=ti.extension.async_mode, async_mode=True)
def test_numpy():
    n = 10000

    @ti.kernel
    def inc(a: ti.ext_arr()):
        for i in range(n):
            a[i] += i

    x = np.zeros(dtype=np.int32, shape=n)
    for i in range(10):
        inc(x)

    for i in range(n):
        assert x[i] == i * 10


@ti.test(require=ti.extension.async_mode, async_mode=True)
def test_listgen_opt_with_offsets():
    x = ti.field(dtype=ti.i32)

    ti.root.pointer(ti.i, 4).dense(ti.i, 4).place(x, offset=-8)

    @ti.kernel
    def inc():
        for i in x:
            x[i] += 1

    for i in range(10):
        inc()

    ti.sync()
    assert ti.get_kernel_stats().get_counters()['launched_tasks_list_gen'] <= 2
