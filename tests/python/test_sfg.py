import numpy as np
import pytest

import taichi as ti


@ti.test(require=[ti.extension.async_mode, ti.extension.sparse],
         async_mode=True)
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


@ti.test(require=ti.extension.async_mode, async_mode=True)
def test_sfg_dead_store_elimination():
    n = 32

    x = ti.field(dtype=float, shape=n, needs_grad=True)
    total_energy = ti.field(dtype=float, shape=(), needs_grad=True)
    unused = ti.field(dtype=float, shape=())

    @ti.kernel
    def gather():
        for i in x:
            e = x[i]**2
            total_energy[None] += e

    @ti.kernel
    def scatter():
        for i in x:
            unused[None] += x[i]

    xnp = np.arange(n, dtype=np.float32)
    x.from_numpy(xnp)
    ti.sync()

    stats = ti.get_kernel_stats()
    stats.clear()

    for _ in range(5):
        with ti.Tape(total_energy):
            gather()
        scatter()

    ti.sync()
    counters = stats.get_counters()

    # gather() should be DSE'ed
    assert counters['sfg_dse_tasks'] > 0

    x_grad = x.grad.to_numpy()
    for i in range(n):
        assert ti.approx(x_grad[i]) == 2.0 * i


@ti.test(require=ti.extension.async_mode, async_mode=True)
def test_global_tmp_value_state():
    # https://github.com/taichi-dev/taichi/issues/2024
    n = 10
    x = ti.field(ti.f32, shape=(n, ))

    @ti.kernel
    def compute_mean_of_boundary_edges() -> ti.i32:
        total = 0.0
        for i in range(n):
            total += x[i] + x[i] * x[i]
        result = total / ti.cast(n, ti.i32)
        return result

    x.from_numpy(np.arange(0, n, dtype=np.float32))
    mean = compute_mean_of_boundary_edges()
    assert ti.approx(mean) == 33
