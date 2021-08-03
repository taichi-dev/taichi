import numpy as np

import taichi as ti


@ti.all_archs
def test_abs():
    x = ti.field(ti.f32)

    N = 16

    ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            x[i] = abs(-i)
            print(x[i])
            ti.static_print(x[i])

    func()

    for i in range(N):
        assert x[i] == i


@ti.all_archs
def test_int():
    x = ti.field(ti.f32)

    N = 16

    ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            x[i] = int(x[i])
            x[i] = float(int(x[i]) // 2)

    for i in range(N):
        x[i] = i + 0.4

    func()

    for i in range(N):
        assert x[i] == i // 2


@ti.all_archs
def test_minmax():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)
    z = ti.field(ti.f32)
    minimum = ti.field(ti.f32)
    maximum = ti.field(ti.f32)

    N = 16

    ti.root.dense(ti.i, N).place(x, y, z, minimum, maximum)

    @ti.kernel
    def func():
        for i in range(N):
            minimum[i] = min(x[i], y[i], z[i])
            maximum[i] = max(x[i], y[i], z[i])

    for i in range(N):
        x[i] = i
        y[i] = N - i
        z[i] = i - 2 if i % 2 else i + 2

    func()

    assert np.allclose(
        minimum.to_numpy(),
        np.minimum(np.minimum(x.to_numpy(), y.to_numpy()), z.to_numpy()))
    assert np.allclose(
        maximum.to_numpy(),
        np.maximum(np.maximum(x.to_numpy(), y.to_numpy()), z.to_numpy()))
