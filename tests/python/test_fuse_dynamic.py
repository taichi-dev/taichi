import time

import pytest

import taichi as ti


def benchmark_fuse_dynamic_x2y2z(size=1024**2, repeat=10, first_n=100):
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    z = ti.field(ti.i32)

    ti.root.dynamic(ti.i, size, chunk_size=2048).place(x, y, z)

    @ti.kernel
    def x_to_y():
        for i in x:
            y[i] = x[i] + 1

    @ti.kernel
    def y_to_z():
        for i in x:
            z[i] = y[i] + 4

    first_n = min(first_n, size)

    for i in range(first_n):
        x[i] = i * 10

    for _ in range(repeat):
        t = time.time()
        x_to_y()
        ti.sync()
        print('x_to_y', time.time() - t)
    print('')

    for _ in range(repeat):
        t = time.time()
        y_to_z()
        ti.sync()
        print('y_to_z', time.time() - t)
    print('')

    for _ in range(repeat):
        t = time.time()
        x_to_y()
        y_to_z()
        ti.sync()
        print('fused x->y->z', time.time() - t)
    print('')

    for i in range(first_n):
        assert x[i] == i * 10
        assert y[i] == x[i] + 1
        assert z[i] == x[i] + 5


@ti.test(require=[ti.extension.async_mode, ti.extension.sparse],
         async_mode=True)
def test_fuse_dynamic_x2y2z():
    benchmark_fuse_dynamic_x2y2z()
