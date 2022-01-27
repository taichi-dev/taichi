import math

import numpy as np
import pytest
from taichi._testing import approx
from taichi.lang.util import has_pytorch

import taichi as ti

archs_support_f16 = [ti.cpu, ti.cuda, ti.vulkan]


@ti.test(arch=archs_support_f16)
def test_snode_read_write():
    dtype = ti.f16
    x = ti.field(dtype, shape=())
    x[None] = 0.3
    print(x[None])
    assert (x[None] == approx(0.3, rel=1e-3))


@ti.test(arch=archs_support_f16)
def test_float16():
    dtype = ti.float16
    x = ti.field(dtype, shape=())
    x[None] = 0.3
    print(x[None])
    assert (x[None] == approx(0.3, rel=1e-3))


@ti.test(arch=archs_support_f16)
def test_to_numpy():
    n = 16
    x = ti.field(ti.f16, shape=n)

    @ti.kernel
    def init():
        for i in x:
            x[i] = i * 2

    init()
    y = x.to_numpy()
    for i in range(n):
        assert (y[i] == 2 * i)


@ti.test(arch=archs_support_f16)
def test_from_numpy():
    n = 16
    y = ti.field(dtype=ti.f16, shape=n)
    x = np.arange(n, dtype=np.half)
    y.from_numpy(x)

    @ti.kernel
    def init():
        for i in y:
            y[i] = 3 * i

    init()
    z = y.to_numpy()
    for i in range(n):
        assert (z[i] == i * 3)


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=archs_support_f16)
def test_to_torch():
    n = 16
    x = ti.field(ti.f16, shape=n)

    @ti.kernel
    def init():
        for i in x:
            x[i] = i * 2

    init()
    y = x.to_torch()
    print(y)
    for i in range(n):
        assert (y[i] == 2 * i)


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=archs_support_f16)
def test_from_torch():
    import torch
    n = 16
    y = ti.field(dtype=ti.f16, shape=n)
    # torch doesn't have rand implementation for float16 so we need to create float first and then convert
    x = torch.range(0, n - 1).to(torch.float16)
    y.from_torch(x)

    @ti.kernel
    def init():
        for i in y:
            y[i] = 3 * i

    init()
    z = y.to_torch()
    for i in range(n):
        assert (z[i] == i * 3)


@ti.test(arch=archs_support_f16)
def test_binary_op():
    dtype = ti.f16
    x = ti.field(dtype, shape=())
    y = ti.field(dtype, shape=())
    z = ti.field(dtype, shape=())

    @ti.kernel
    def add():
        x[None] = y[None] + z[None]
        x[None] = x[None] * z[None]

    y[None] = 0.2
    z[None] = 0.72
    add()
    u = x.to_numpy()
    assert (u[None] == approx(0.6624, rel=1e-3))


@ti.test(arch=archs_support_f16)
def test_rand_promote():
    dtype = ti.f16
    x = ti.field(dtype, shape=(4, 4))

    @ti.kernel
    def init():
        for i, j in x:
            x[i, j] = ti.random(dtype=dtype)
            print(x[i, j])

    init()


@ti.test(arch=archs_support_f16)
def test_unary_op():
    dtype = ti.f16
    x = ti.field(dtype, shape=())
    y = ti.field(dtype, shape=())

    @ti.kernel
    def foo():
        x[None] = -y[None]
        x[None] = ti.floor(x[None])
        y[None] = ti.ceil(y[None])

    y[None] = -1.4
    foo()
    assert (x[None] == approx(1, rel=1e-3))
    assert (y[None] == approx(-1, rel=1e-3))


@ti.test(arch=archs_support_f16)
def test_extra_unary_promote():
    dtype = ti.f16
    x = ti.field(dtype, shape=())
    y = ti.field(dtype, shape=())

    @ti.kernel
    def foo():
        x[None] = abs(y[None])

    y[None] = -0.3
    foo()
    assert (x[None] == approx(0.3, rel=1e-3))


@ti.test(arch=archs_support_f16)
def test_binary_extra_promote():
    x = ti.field(dtype=ti.f16, shape=())
    y = ti.field(dtype=ti.f16, shape=())
    z = ti.field(dtype=ti.f16, shape=())

    @ti.kernel
    def foo():
        y[None] = x[None]**2
        z[None] = ti.atan2(y[None], 0.3)

    x[None] = 0.1
    foo()
    assert (z[None] == approx(math.atan2(0.1**2, 0.3), rel=1e-3))


@ti.test(arch=archs_support_f16)
def test_arg_f16():
    dtype = ti.f16
    x = ti.field(dtype, shape=())
    y = ti.field(dtype, shape=())

    @ti.kernel
    def foo(a: ti.f16):
        x[None] = y[None] + a

    y[None] = -0.3
    foo(1.2)
    assert (x[None] == approx(0.9, rel=1e-3))


@ti.test(arch=archs_support_f16)
def test_fractal_f16():
    n = 320
    pixels = ti.field(dtype=ti.f16, shape=(n * 2, n))

    @ti.func
    def complex_sqr(z):
        return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2], dt=ti.f16)

    @ti.kernel
    def paint(t: float):
        for i, j in pixels:  # Parallelized over all pixels
            c = ti.Vector([-0.8, ti.cos(t) * 0.2], dt=ti.f16)
            z = ti.Vector([i / n - 1, j / n - 0.5], dt=ti.f16) * 2
            iterations = 0
            while z.norm() < 20 and iterations < 50:
                z = complex_sqr(z) + c
                iterations += 1
            pixels[i, j] = 1 - iterations * 0.02

    paint(0.03)


# TODO(): Vulkan support
@ti.test(arch=[ti.cpu, ti.cuda])
def test_atomic_add_f16():
    f = ti.field(dtype=ti.f16, shape=(2))

    @ti.kernel
    def foo():
        # Parallel sum
        for i in range(1000):
            f[0] += 1.12

        # Serial sum
        for _ in range(1):
            for i in range(1000):
                f[1] = f[1] + 1.12

    foo()
    assert (f[0] == approx(f[1], rel=1e-3))


# TODO(): Vulkan support
@ti.test(arch=[ti.cpu, ti.cuda])
def test_atomic_max_f16():
    f = ti.field(dtype=ti.f16, shape=(2))

    @ti.kernel
    def foo():
        # Parallel max
        for i in range(1000):
            ti.atomic_max(f[0], 1.12 * i)

        # Serial max
        for _ in range(1):
            for i in range(1000):
                f[1] = ti.max(1.12 * i, f[1])

    foo()
    assert (f[0] == approx(f[1], rel=1e-3))


# TODO(): Vulkan support
@ti.test(arch=[ti.cpu, ti.cuda])
def test_atomic_min_f16():
    f = ti.field(dtype=ti.f16, shape=(2))

    @ti.kernel
    def foo():
        # Parallel min
        for i in range(1000):
            ti.atomic_min(f[0], -3.13 * i)

        # Serial min
        for _ in range(1):
            for i in range(1000):
                f[1] = ti.min(-3.13 * i, f[1])

    foo()
    assert (f[0] == approx(f[1], rel=1e-3))


@ti.test(arch=archs_support_f16)
def test_cast_f32_to_f16():
    @ti.kernel
    def func() -> ti.f16:
        a = ti.cast(23.0, ti.f32)
        b = ti.cast(4.0, ti.f32)
        return ti.cast(a * b, ti.f16)

    assert func() == pytest.approx(23.0 * 4.0, 1e-4)


@ti.test(arch=archs_support_f16, require=ti.extension.data64)
def test_cast_f64_to_f16():
    @ti.kernel
    def func() -> ti.f16:
        a = ti.cast(23.0, ti.f64)
        b = ti.cast(4.0, ti.f64)
        return ti.cast(a * b, ti.f16)

    assert func() == pytest.approx(23.0 * 4.0, 1e-4)
