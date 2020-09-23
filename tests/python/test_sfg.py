import taichi as ti
import pytest


@ti.test(require=ti.extension.async_mode, async_mode=True)
def test_remove_clear_list_from_fused_serial():
    # ti.init(ti.cpu, async_mode=True, async_opt_intermediate_file='tmp/remove_clear_list')
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    z = ti.field(ti.i32, shape=())

    n = 32
    ti.root.bitmasked(ti.i, n).place(x)
    ti.root.bitmasked(ti.i, n).place(y)

    @ti.kernel
    def init_xy():
        for i in range(n):
            if i & 1:
                x[i] = i
            else:
                y[i] = i

    init_xy()
    ti.sync()

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

    xs = x.to_numpy()
    ys = y.to_numpy()
    for i in range(n):
        if i & 1:
            assert xs[i] == i + 2
            assert ys[i] == 0
        else:
            assert ys[i] == i + 2
            assert xs[i] == 0
