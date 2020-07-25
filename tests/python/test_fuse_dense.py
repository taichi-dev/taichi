import taichi as ti
import time
import pytest


def benchmark_fuse_dense_x2y2z(size=1024**3, repeat=10, first_n=100):
    # TODO: this can also be made as a benchmark or a regression test
    x = ti.var(ti.i32, shape=(size, ))
    y = ti.var(ti.i32, shape=(size, ))
    z = ti.var(ti.i32, shape=(size, ))
    first_n = min(first_n, size)

    @ti.kernel
    def x_to_y():
        for i in x:
            y[i] = x[i] + 1

    @ti.kernel
    def y_to_z():
        for i in x:
            z[i] = y[i] + 4

    for i in range(first_n):
        x[i] = i * 10

    for _ in range(repeat):
        t = time.time()
        x_to_y()
        ti.sync()
        print('x_to_y', time.time() - t)

    for _ in range(repeat):
        t = time.time()
        y_to_z()
        ti.sync()
        print('y_to_z', time.time() - t)

    for _ in range(repeat):
        t = time.time()
        x_to_y()
        y_to_z()
        ti.sync()
        print('fused x->y->z', time.time() - t)

    for i in range(first_n):
        assert x[i] == i * 10
        assert y[i] == x[i] + 1
        assert z[i] == x[i] + 5


def benchmark_fuse_reduction(size=1024**3, repeat=10, first_n=100):
    # TODO: this can also be made as a benchmark or a regression test
    x = ti.var(ti.i32, shape=(size, ))
    first_n = min(first_n, size)

    @ti.kernel
    def reset():
        for i in range(first_n):
            x[i] = i * 10

    @ti.kernel
    def inc():
        for i in x:
            x[i] = x[i] + 1

    reset()
    ti.sync()
    for _ in range(repeat):
        t = time.time()
        inc()
        ti.sync()
        print('single inc', time.time() - t)

    reset()
    ti.sync()
    t = time.time()
    for _ in range(repeat):
        inc()
    ti.sync()
    duration = time.time() - t
    print(f'fused {repeat} inc: total={duration} average={duration / repeat}')

    for i in range(first_n):
        assert x[i] == i * 10 + repeat


@ti.archs_with([ti.cpu], async_mode=True)
def test_fuse_dense_x2y2z():
    benchmark_fuse_dense_x2y2z(size=100 * 1024**2)


@ti.archs_with([ti.cpu], async_mode=True)
def test_fuse_reduction():
    benchmark_fuse_reduction(size=10 * 1024**2)


@ti.archs_with([ti.cpu], async_mode=True)
def test_no_fuse_sigs_mismatch():
    n = 4096
    x = ti.field(ti.i32, shape=(n, ))

    @ti.kernel
    def inc_i():
        for i in x:
            x[i] += i

    @ti.kernel
    def inc_by(k: ti.i32):
        for i in x:
            x[i] += k

    repeat = 5
    for i in range(repeat):
        inc_i()
        inc_by(i)

    x = x.to_numpy()
    for i in range(n):
        assert x[i] == i * repeat + ((repeat - 1) * repeat // 2)
