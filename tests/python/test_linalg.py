import math

import numpy as np
import pytest

import taichi as ti
from taichi import approx


@ti.test()
def test_const_init():
    a = ti.Matrix.field(2, 3, dtype=ti.i32, shape=())
    b = ti.Vector.field(3, dtype=ti.i32, shape=())

    @ti.kernel
    def init():
        a[None] = ti.Matrix([[0, 1, 2], [3, 4, 5]])
        b[None] = ti.Vector([0, 1, 2])

    init()

    for i in range(2):
        for j in range(3):
            assert a[None][i, j] == i * 3 + j

    for j in range(3):
        assert b[None][j] == j


@ti.test()
def test_basic_utils():
    a = ti.Vector.field(3, dtype=ti.f32)
    b = ti.Vector.field(2, dtype=ti.f32)
    abT = ti.Matrix.field(3, 2, dtype=ti.f32)
    aNormalized = ti.Vector.field(3, dtype=ti.f32)

    normA = ti.field(ti.f32)
    normSqrA = ti.field(ti.f32)
    normInvA = ti.field(ti.f32)

    ti.root.place(a, b, abT, aNormalized, normA, normSqrA, normInvA)

    @ti.kernel
    def init():
        a[None] = ti.Vector([1.0, 2.0, 3.0])
        b[None] = ti.Vector([4.0, 5.0])
        abT[None] = a[None].outer_product(b[None])

        normA[None] = a[None].norm()
        normSqrA[None] = a[None].norm_sqr()
        normInvA[None] = a[None].norm_inv()

        aNormalized[None] = a[None].normalized()

    init()

    for i in range(3):
        for j in range(2):
            assert abT[None][i, j] == a[None][i] * b[None][j]

    sqrt14 = np.sqrt(14.0)
    invSqrt14 = 1.0 / sqrt14
    assert normSqrA[None] == approx(14.0)
    assert normInvA[None] == approx(invSqrt14)
    assert normA[None] == approx(sqrt14)
    assert aNormalized[None][0] == approx(1.0 * invSqrt14)
    assert aNormalized[None][1] == approx(2.0 * invSqrt14)
    assert aNormalized[None][2] == approx(3.0 * invSqrt14)


@ti.test()
def test_cross():
    a = ti.Vector.field(3, dtype=ti.f32)
    b = ti.Vector.field(3, dtype=ti.f32)
    c = ti.Vector.field(3, dtype=ti.f32)

    a2 = ti.Vector.field(2, dtype=ti.f32)
    b2 = ti.Vector.field(2, dtype=ti.f32)
    c2 = ti.field(dtype=ti.f32)

    ti.root.place(a, b, c, a2, b2, c2)

    @ti.kernel
    def init():
        a[None] = ti.Vector([1.0, 2.0, 3.0])
        b[None] = ti.Vector([4.0, 5.0, 6.0])
        c[None] = a[None].cross(b[None])

        a2[None] = ti.Vector([1.0, 2.0])
        b2[None] = ti.Vector([4.0, 5.0])
        c2[None] = a2[None].cross(b2[None])

    init()
    assert c[None][0] == -3.0
    assert c[None][1] == 6.0
    assert c[None][2] == -3.0
    assert c2[None] == -3.0


@ti.test()
def test_dot():
    a = ti.Vector.field(3, dtype=ti.f32)
    b = ti.Vector.field(3, dtype=ti.f32)
    c = ti.field(dtype=ti.f32)

    a2 = ti.Vector.field(2, dtype=ti.f32)
    b2 = ti.Vector.field(2, dtype=ti.f32)
    c2 = ti.field(dtype=ti.f32)

    ti.root.place(a, b, c, a2, b2, c2)

    @ti.kernel
    def init():
        a[None] = ti.Vector([1.0, 2.0, 3.0])
        b[None] = ti.Vector([4.0, 5.0, 6.0])
        c[None] = a[None].dot(b[None])

        a2[None] = ti.Vector([1.0, 2.0])
        b2[None] = ti.Vector([4.0, 5.0])
        c2[None] = a2[None].dot(b2[None])

    init()
    assert c[None] == 32.0
    assert c2[None] == 14.0


@ti.test()
def test_transpose():
    dim = 3
    m = ti.Matrix.field(dim, dim, ti.f32)

    ti.root.place(m)

    @ti.kernel
    def transpose():
        mat = m[None].transpose()
        m[None] = mat

    for i in range(dim):
        for j in range(dim):
            m[None][i, j] = i * 2 + j * 7

    transpose()

    for i in range(dim):
        for j in range(dim):
            assert m[None][j, i] == approx(i * 2 + j * 7)


def _test_polar_decomp(dim, dt):
    m = ti.Matrix.field(dim, dim, dt)
    r = ti.Matrix.field(dim, dim, dt)
    s = ti.Matrix.field(dim, dim, dt)
    I = ti.Matrix.field(dim, dim, dt)
    D = ti.Matrix.field(dim, dim, dt)

    ti.root.place(m, r, s, I, D)

    @ti.kernel
    def polar():
        R, S = ti.polar_decompose(m[None], dt)
        r[None] = R
        s[None] = S
        m[None] = R @ S
        I[None] = R @ R.transpose()
        D[None] = S - S.transpose()

    def V(i, j):
        return i * 2 + j * 7 + int(i == j) * 3

    for i in range(dim):
        for j in range(dim):
            m[None][i, j] = V(i, j)

    polar()

    tol = 5e-5 if dt == ti.f32 else 1e-12

    for i in range(dim):
        for j in range(dim):
            assert m[None][i, j] == approx(V(i, j), abs=tol)
            assert I[None][i, j] == approx(int(i == j), abs=tol)
            assert D[None][i, j] == approx(0, abs=tol)


def test_polar_decomp():
    for dim in [2, 3]:
        for dt in [ti.f32, ti.f64]:

            @ti.test(require=ti.extension.data64 if dt == ti.f64 else [],
                     default_fp=dt)
            def wrapped():
                _test_polar_decomp(dim, dt)

            wrapped()


@ti.test()
def test_matrix():
    x = ti.Matrix.field(2, 2, dtype=ti.i32)

    ti.root.dense(ti.i, 16).place(x)

    @ti.kernel
    def inc():
        for i in x:
            delta = ti.Matrix([[3, 0], [0, 0]])
            x[i][1, 1] = x[i][0, 0] + 1
            x[i] = x[i] + delta
            x[i] += delta

    for i in range(10):
        x[i][0, 0] = i

    inc()

    for i in range(10):
        assert x[i][0, 0] == 6 + i
        assert x[i][1, 1] == 1 + i


@ti.test()
def _test_mat_inverse_size(n):
    m = ti.Matrix.field(n, n, dtype=ti.f32, shape=())
    M = np.empty(shape=(n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            M[i, j] = i * j + i * 3 + j + 1 + int(i == j) * 4
    assert np.linalg.det(M) != 0

    m.from_numpy(M)

    @ti.kernel
    def invert():
        m[None] = m[None].inverse()

    invert()

    m_np = m.to_numpy(keep_dims=True)
    np.testing.assert_almost_equal(m_np, np.linalg.inv(M))


def test_mat_inverse():
    for n in range(1, 5):
        _test_mat_inverse_size(n)


@ti.test()
def test_matrix_factories():
    a = ti.Vector.field(3, dtype=ti.i32, shape=3)
    b = ti.Matrix.field(2, 2, dtype=ti.f32, shape=2)
    c = ti.Matrix.field(2, 3, dtype=ti.f32, shape=2)

    @ti.kernel
    def fill():
        b[0] = ti.Matrix.identity(ti.f32, 2)
        b[1] = ti.Matrix.rotation2d(math.pi / 3)
        c[0] = ti.Matrix.zero(ti.f32, 2, 3)
        c[1] = ti.Matrix.one(ti.f32, 2, 3)
        for i in ti.static(range(3)):
            a[i] = ti.Vector.unit(3, i)

    fill()

    for i in range(3):
        for j in range(3):
            assert a[i][j] == int(i == j)

    sqrt3o2 = math.sqrt(3) / 2
    assert b[0].value.to_numpy() == approx(np.eye(2))
    assert b[1].value.to_numpy() == approx(
        np.array([[0.5, -sqrt3o2], [sqrt3o2, 0.5]]))
    assert c[0].value.to_numpy() == approx(np.zeros((2, 3)))
    assert c[1].value.to_numpy() == approx(np.ones((2, 3)))


# TODO: move codes below to test_matrix.py:


@ti.test()
def test_init_matrix_from_vectors():
    m1 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(3))
    m2 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(3))
    m3 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(3))
    m4 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(3))

    @ti.kernel
    def fill():
        for i in range(3):
            a = ti.Vector([1.0, 4.0, 7.0])
            b = ti.Vector([2.0, 5.0, 8.0])
            c = ti.Vector([3.0, 6.0, 9.0])
            m1[i] = ti.Matrix.rows([a, b, c])
            m2[i] = ti.Matrix.cols([a, b, c])
            m3[i] = ti.Matrix.rows([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0],
                                    [3.0, 6.0, 9.0]])
            m4[i] = ti.Matrix.cols([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0],
                                    [3.0, 6.0, 9.0]])

    fill()

    for j in range(3):
        for i in range(3):
            assert m1[0][i, j] == int(i + 3 * j + 1)
            assert m2[0][j, i] == int(i + 3 * j + 1)
            assert m3[0][i, j] == int(i + 3 * j + 1)
            assert m4[0][j, i] == int(i + 3 * j + 1)


# TODO: Remove this once the APIs are obsolete.
@pytest.mark.filterwarnings('ignore')
@ti.test(arch=ti.get_host_arch_list())
def test_init_matrix_from_vectors_deprecated():
    m1 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(3))
    m2 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(3))
    m3 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(3))
    m4 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(3))

    @ti.kernel
    def fill():
        for i in range(3):
            a = ti.Vector([1.0, 4.0, 7.0])
            b = ti.Vector([2.0, 5.0, 8.0])
            c = ti.Vector([3.0, 6.0, 9.0])
            m1[i] = ti.Matrix.rows([a, b, c])
            m2[i] = ti.Matrix.cols([a, b, c])
            m3[i] = ti.Matrix.rows([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0],
                                    [3.0, 6.0, 9.0]])
            m4[i] = ti.Matrix.cols([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0],
                                    [3.0, 6.0, 9.0]])

    fill()

    for j in range(3):
        for i in range(3):
            assert m1[0][i, j] == int(i + 3 * j + 1)
            assert m2[0][j, i] == int(i + 3 * j + 1)
            assert m3[0][i, j] == int(i + 3 * j + 1)
            assert m4[0][j, i] == int(i + 3 * j + 1)


@pytest.mark.filterwarnings('ignore')
@ti.test(arch=ti.get_host_arch_list())
def test_to_numpy_as_vector_deprecated():
    v = ti.Vector.field(3, dtype=ti.f32, shape=(2))
    u = np.array([[2, 3, 4], [5, 6, 7]])
    v.from_numpy(u)
    assert v.to_numpy(as_vector=True) == approx(u)
    assert v.to_numpy() == approx(u)


@ti.test()
def test_any_all():
    a = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    b = ti.field(dtype=ti.i32, shape=())
    c = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def func():
        b[None] = any(a[None])
        c[None] = all(a[None])

    for i in range(2):
        for j in range(2):
            a[None][0, 0] = i
            a[None][1, 0] = j
            a[None][1, 1] = i
            a[None][0, 1] = j

            func()
            if i == 1 or j == 1:
                assert b[None] == 1
            else:
                assert b[None] == 0

            if i == 1 and j == 1:
                assert c[None] == 1
            else:
                assert c[None] == 0


@ti.test()
def test_min_max():
    a = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    b = ti.field(dtype=ti.i32, shape=())
    c = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def func():
        b[None] = a[None].max()
        c[None] = a[None].min()

    for i in range(2):
        for j in range(2):
            a[None][0, 0] = i
            a[None][1, 0] = j
            a[None][1, 1] = i
            a[None][0, 1] = j

            func()
            assert b[None] == max(i, j)
            assert c[None] == min(i, j)


@ti.test(exclude=[ti.vulkan, ti.opengl, ti.cc])
def test_min():
    @ti.kernel
    def min_u16(a: ti.u16, b: ti.u16) -> ti.u16:
        return ti.min(a, b)

    @ti.kernel
    def min_i16(a: ti.i16, b: ti.i16) -> ti.i16:
        return ti.min(a, b)

    @ti.kernel
    def min_u32(a: ti.u32, b: ti.u32) -> ti.u32:
        return ti.min(a, b)

    @ti.kernel
    def min_i32(a: ti.i32, b: ti.i32) -> ti.i32:
        return ti.min(a, b)

    a, b = 1, 2
    assert min_u16(a, b) == min(a, b)
    assert min_i16(a, b) == min(a, b)
    assert min_u32(a, b) == min(a, b)
    assert min_i32(a, b) == min(a, b)


@ti.test(exclude=[ti.vulkan, ti.opengl, ti.cc])
def test_max():
    @ti.kernel
    def max_u16(a: ti.u16, b: ti.u16) -> ti.u16:
        return ti.max(a, b)

    @ti.kernel
    def max_i16(a: ti.i16, b: ti.i16) -> ti.i16:
        return ti.max(a, b)

    @ti.kernel
    def max_u32(a: ti.u32, b: ti.u32) -> ti.u32:
        return ti.max(a, b)

    @ti.kernel
    def max_i32(a: ti.i32, b: ti.i32) -> ti.i32:
        return ti.max(a, b)

    a, b = 1, 2
    assert max_u16(a, b) == max(a, b)
    assert max_i16(a, b) == max(a, b)
    assert max_u32(a, b) == max(a, b)
    assert max_i32(a, b) == max(a, b)


# must not throw any error:
@ti.test()
def test_matrix_list_assign():

    m = ti.Matrix.field(2, 2, dtype=ti.i32, shape=(2, 2, 1))
    v = ti.Vector.field(2, dtype=ti.i32, shape=(2, 2, 1))

    m[1, 0, 0] = [[4, 3], [6, 7]]
    v[1, 0, 0] = [8, 4]

    assert np.allclose(m.to_numpy()[1, 0, 0, :, :], np.array([[4, 3], [6, 7]]))
    assert np.allclose(v.to_numpy()[1, 0, 0, :], np.array([8, 4]))

    @ti.kernel
    def func():
        m[1, 0, 0] = [[1, 2], [3, 4]]
        v[1, 0, 0] = [5, 6]
        m[1, 0, 0] += [[1, 2], [3, 4]]
        v[1, 0, 0] += [5, 6]

    func()
    assert np.allclose(m.to_numpy()[1, 0, 0, :, :], np.array([[2, 4], [6, 8]]))
    assert np.allclose(v.to_numpy()[1, 0, 0, :], np.array([10, 12]))


@ti.test(arch=ti.get_host_arch_list())
def test_vector_xyzw_accessor():
    u = ti.Vector.field(2, dtype=ti.i32, shape=(2, 2, 1))
    v = ti.Vector.field(4, dtype=ti.i32, shape=(2, 2, 1))

    u[1, 0, 0].y = 3
    v[1, 0, 0].z = 0
    v[1, 0, 0].w = 4

    @ti.kernel
    def func():
        u[1, 0, 0].x = 8 * u[1, 0, 0].y
        v[1, 0, 0].z = 1 - v[1, 0, 0].w
        v[1, 0, 0].x = 6

    func()
    assert u[1, 0, 0].x == 24
    assert u[1, 0, 0].y == 3
    assert v[1, 0, 0].z == -3
    assert v[1, 0, 0].w == 4
    assert np.allclose(v.to_numpy()[1, 0, 0, :], np.array([6, 0, -3, 4]))


@ti.test(arch=ti.get_host_arch_list())
def test_diag():
    m1 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

    @ti.kernel
    def fill():
        m1[None] = ti.Matrix.diag(dim=3, val=1.4)

    fill()

    for i in range(3):
        for j in range(3):
            if i == j:
                assert m1[None][i, j] == approx(1.4)
            else:
                assert m1[None][i, j] == 0.0
