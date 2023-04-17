import pytest
from taichi.lang.exception import TaichiCompilationError

import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_dynamic():
    x = ti.field(ti.f32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        pass

    for i in range(n):
        x[i] = i

    for i in range(n):
        assert x[i] == i


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_dynamic2():
    x = ti.field(ti.f32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            x[i] = i

    func()

    for i in range(n):
        assert x[i] == i


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_dynamic_matrix():
    x = ti.Matrix.field(2, 1, dtype=ti.i32)
    n = 8192

    ti.root.dynamic(ti.i, n, chunk_size=128).place(x)

    @ti.kernel
    def func():
        ti.loop_config(serialize=True)
        for i in range(n // 4):
            x[i * 4][1, 0] = i

    func()

    for i in range(n // 4):
        a = x[i * 4][1, 0]
        assert a == i
        if i + 1 < n // 4:
            b = x[i * 4 + 1][1, 0]
            assert b == 0


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_append():
    x = ti.field(ti.i32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            ti.append(x.parent(), [], i)

    func()

    elements = []
    for i in range(n):
        elements.append(x[i])
    elements.sort()
    for i in range(n):
        assert elements[i] == i


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_length():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32, shape=())
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            ti.append(x.parent(), [], i)

    func()

    @ti.kernel
    def get_len():
        y[None] = ti.length(x.parent(), [])

    get_len()

    assert y[None] == n


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_append_ret_value():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    z = ti.field(ti.i32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)
    ti.root.dynamic(ti.i, n, 32).place(y)
    ti.root.dynamic(ti.i, n, 32).place(z)

    @ti.kernel
    def func():
        for i in range(n):
            u = ti.append(x.parent(), [], i)
            y[u] = i + 1
            z[u] = i + 3

    func()

    for i in range(n):
        assert x[i] + 1 == y[i]
        assert x[i] + 3 == z[i]


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_dense_dynamic():
    n = 128
    x = ti.field(ti.i32)
    l = ti.field(ti.i32, shape=n)

    ti.root.dense(ti.i, n).dynamic(ti.j, n, 8).place(x)

    @ti.kernel
    def func():
        ti.loop_config(serialize=True)
        for i in range(n):
            for j in range(n):
                ti.append(x.parent(), j, i)

        for i in range(n):
            l[i] = ti.length(x.parent(), i)

    func()

    for i in range(n):
        assert l[i] == n


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_dense_dynamic_len():
    n = 128
    x = ti.field(ti.i32)
    l = ti.field(ti.i32, shape=n)

    ti.root.dense(ti.i, n).dynamic(ti.j, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            l[i] = ti.length(x.parent(), i)

    func()

    for i in range(n):
        assert l[i] == 0


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_dynamic_activate():
    # record the lengths
    l = ti.field(ti.i32, 3)
    x = ti.field(ti.i32)
    xp = ti.root.dynamic(ti.i, 32, 32)
    xp.place(x)

    m = 5

    @ti.kernel
    def func():
        for i in range(m):
            ti.append(xp, [], i)
        l[0] = ti.length(xp, [])
        x[20] = 42
        l[1] = ti.length(xp, [])
        x[10] = 43
        l[2] = ti.length(xp, [])

    func()
    l = l.to_numpy()
    assert l[0] == m
    assert l[1] == 21
    assert l[2] == 21


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_append_u8():
    x = ti.field(ti.u8)
    pixel = ti.root.dynamic(ti.j, 20)
    pixel.place(x)

    @ti.kernel
    def make_list():
        ti.loop_config(serialize=True)
        for i in range(20):
            x[()].append(i * i * i)

    make_list()

    for i in range(20):
        assert x[i] == i * i * i % 256


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_append_u64():
    x = ti.field(ti.u64)
    pixel = ti.root.dynamic(ti.i, 20)
    pixel.place(x)

    @ti.kernel
    def make_list():
        ti.loop_config(serialize=True)
        for i in range(20):
            x[()].append(i * i * i * ti.u64(10000000000))

    make_list()

    for i in range(20):
        assert x[i] == i * i * i * 10000000000


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_append_struct():
    struct = ti.types.struct(a=ti.u8, b=ti.u16, c=ti.u32, d=ti.u64)
    x = struct.field()
    pixel = ti.root.dense(ti.i, 10).dynamic(ti.j, 20, 5)
    pixel.place(x)

    @ti.kernel
    def make_list():
        for i in range(10):
            for j in range(20):
                x[i].append(
                    struct(
                        i * j * 10,
                        i * j * 10000,
                        i * j * 100000000,
                        i * j * ti.u64(10000000000),
                    )
                )

    make_list()

    for i in range(10):
        for j in range(20):
            assert x[i, j].a == i * j * 10 % 256
            assert x[i, j].b == i * j * 10000 % 65536
            assert x[i, j].c == i * j * 100000000 % 4294967296
            assert x[i, j].d == i * j * 10000000000


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_append_matrix():
    mat = ti.types.matrix(n=2, m=2, dtype=ti.u8)
    f = mat.field()
    pixel = ti.root.dense(ti.i, 10).dynamic(ti.j, 20, 4)
    pixel.place(f)

    @ti.kernel
    def make_list():
        for i in range(10):
            for j in range(20):
                f[i].append(ti.Matrix([[i * j, i * j * 2], [i * j * 3, i * j * 4]], dt=ti.u8))

    make_list()

    for i in range(10):
        for j in range(20):
            for k in range(4):
                assert f[i, j][k // 2, k % 2] == i * j * (k + 1) % 256


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_append_matrix_in_struct():
    mat = ti.types.matrix(n=2, m=2, dtype=ti.u8)
    struct = ti.types.struct(a=ti.u64, b=mat, c=ti.u16)
    f = struct.field()
    pixel = ti.root.dense(ti.i, 10).dynamic(ti.j, 20, 4)
    pixel.place(f)

    @ti.kernel
    def make_list():
        for i in range(10):
            for j in range(20):
                f[i].append(
                    struct(
                        i * j * ti.u64(10**10),
                        ti.Matrix([[i * j, i * j * 2], [i * j * 3, i * j * 4]], dt=ti.u8),
                        i * j * 5000,
                    )
                )

    make_list()

    for i in range(10):
        for j in range(20):
            assert f[i, j].a == i * j * (10**10)
            for k in range(4):
                assert f[i, j].b[k // 2, k % 2] == i * j * (k + 1) % 256
            assert f[i, j].c == i * j * 5000 % 65536
