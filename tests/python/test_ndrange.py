import numpy as np

import taichi as ti


@ti.all_archs
def test_1d():
    x = ti.field(ti.f32, shape=(16))

    @ti.kernel
    def func():
        for i in ti.ndrange((4, 10)):
            x[i] = i

    func()

    for i in range(16):
        if 4 <= i < 10:
            assert x[i] == i
        else:
            assert x[i] == 0


@ti.all_archs
def test_2d():
    x = ti.field(ti.f32, shape=(16, 32))

    t = 8

    @ti.kernel
    def func():
        for i, j in ti.ndrange((4, 10), (3, t)):
            val = i + j * 10
            x[i, j] = val

    func()
    for i in range(16):
        for j in range(32):
            if 4 <= i < 10 and 3 <= j < 8:
                assert x[i, j] == i + j * 10
            else:
                assert x[i, j] == 0


@ti.all_archs
def test_3d():
    x = ti.field(ti.f32, shape=(16, 32, 64))

    @ti.kernel
    def func():
        for i, j, k in ti.ndrange((4, 10), (3, 8), 17):
            x[i, j, k] = i + j * 10 + k * 100

    func()
    for i in range(16):
        for j in range(32):
            for k in range(64):
                if 4 <= i < 10 and 3 <= j < 8 and k < 17:
                    assert x[i, j, k] == i + j * 10 + k * 100
                else:
                    assert x[i, j, k] == 0


@ti.all_archs
def test_static_grouped():
    x = ti.field(ti.f32, shape=(16, 32, 64))

    @ti.kernel
    def func():
        for I in ti.static(ti.grouped(ti.ndrange((4, 5), (3, 5), 5))):
            x[I] = I[0] + I[1] * 10 + I[2] * 100

    func()
    for i in range(16):
        for j in range(32):
            for k in range(64):
                if 4 <= i < 5 and 3 <= j < 5 and k < 5:
                    assert x[i, j, k] == i + j * 10 + k * 100
                else:
                    assert x[i, j, k] == 0


@ti.all_archs
def test_static_grouped_static():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=(16, 4))

    @ti.kernel
    def func():
        for i, j in ti.ndrange(16, 4):
            for I in ti.static(ti.grouped(ti.ndrange(2, 3))):
                x[i, j][I] = I[0] + I[1] * 10 + i + j * 4

    func()
    for i in range(16):
        for j in range(4):
            for k in range(2):
                for l in range(3):
                    assert x[i, j][k, l] == k + l * 10 + i + j * 4


@ti.all_archs
def test_field_init_eye():
    # https://github.com/taichi-dev/taichi/issues/1824

    n = 32

    A = ti.field(ti.f32, (n, n))

    @ti.kernel
    def init():
        for i, j in ti.ndrange(n, n):
            if i == j:
                A[i, j] = 1

    init()
    assert np.allclose(A.to_numpy(), np.eye(n, dtype=np.float32))


@ti.all_archs
def test_ndrange_index_floordiv():
    # https://github.com/taichi-dev/taichi/issues/1829

    n = 10

    A = ti.field(ti.f32, (n, n))

    @ti.kernel
    def init():
        for i, j in ti.ndrange(n, n):
            if i // 2 == 0:
                A[i, j] = i

    init()
    for i in range(n):
        for j in range(n):
            if i // 2 == 0:
                assert A[i, j] == i
            else:
                assert A[i, j] == 0


@ti.test()
def test_nested_ndrange():
    # https://github.com/taichi-dev/taichi/issues/1829

    n = 2

    A = ti.field(ti.i32, (n, n, n, n))

    @ti.kernel
    def init():
        for i, j in ti.ndrange(n, n):
            for k, l in ti.ndrange(n, n):
                r = i * n**3 + j * n**2 + k * n + l
                A[i, j, k, l] = r

    init()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    r = i * n**3 + j * n**2 + k * n + l
                    assert A[i, j, k, l] == r


@ti.test(ti.cpu)
def test_ndrange_ast_transform():
    n, u, v = 4, 3, 2

    a = ti.field(ti.i32, ())
    b = ti.field(ti.i32, ())
    A = ti.field(ti.i32, (n, n))

    @ti.kernel
    def func():
        # `__getitem__ cannot be called from Python-scope` will be raised if
        # `a[None]` is not transformed to `ti.subscript(a, None)` in ti.ndrange:
        for i, j in ti.ndrange(a[None], b[None]):
            r = i * n + j + 1
            A[i, j] = r

    a[None] = u
    b[None] = v

    func()

    for i in range(n):
        for j in range(n):
            if i < u and j < v:
                r = i * n + j + 1
            else:
                r = 0
            assert A[i, j] == r
