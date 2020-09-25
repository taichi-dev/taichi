import taichi as ti
import pytest


@ti.test(require=ti.extension.async_mode, async_mode=True)
def test_remove_clear_list_from_fused_serial():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    z = ti.field(ti.i32, shape=())

    n = 32
    ti.root.pointer(ti.i, n).dense(ti.i, 1).place(x)
    ti.root.pointer(ti.i, n).dense(ti.i, 1).place(y)

    @ti.kernel
    def init_xy():
        for i in range(n):
            if i & 1:
                x[i] = i
            else:
                y[i] = i

    init_xy()
    ti.sync()

    stats = ti.get_kernel_stats()
    stats.clear()

    @ti.kernel
    def inc(f: ti.template()):
        for i in f:
            f[i] += 1

    @ti.kernel
    def serial_z():
        z[None] = 40
        z[None] += 2

    inc(x)
    inc(y)
    serial_z()
    inc(x)
    inc(y)
    ti.sync()

    counters = stats.get_counters()
    # each of x and y has two listgens: root -> pointer -> dense
    assert int(counters['launched_tasks_list_gen']) == 4
    # clear list tasks have been fused into serial_z
    assert int(counters['launched_tasks_serial']) == 1

    xs = x.to_numpy()
    ys = y.to_numpy()
    for i in range(n):
        if i & 1:
            assert xs[i] == i + 2
            assert ys[i] == 0
        else:
            assert ys[i] == i + 2
            assert xs[i] == 0
