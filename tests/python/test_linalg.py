import taichi as ti
import numpy as np
from taichi import approx


@ti.all_archs
def test_transpose():
    dim = 3
    m = ti.Matrix(dim, dim, ti.f32)

    @ti.layout
    def place():
        ti.root.place(m)

    @ti.kernel
    def transpose():
        mat = ti.transposed(m[None])
        m[None] = mat

    for i in range(dim):
        for j in range(dim):
            m(i, j)[None] = i * 2 + j * 7

    transpose()

    for i in range(dim):
        for j in range(dim):
            assert m(j, i)[None] == approx(i * 2 + j * 7)


def _test_polar_decomp(dim, dt):
    m = ti.Matrix(dim, dim, dt)
    r = ti.Matrix(dim, dim, dt)
    s = ti.Matrix(dim, dim, dt)
    I = ti.Matrix(dim, dim, dt)
    D = ti.Matrix(dim, dim, dt)

    @ti.layout
    def place():
        ti.root.place(m, r, s, I, D)

    @ti.kernel
    def polar():
        R, S = ti.polar_decompose(m[None], dt)
        r[None] = R
        s[None] = S
        m[None] = R @ S
        I[None] = R @ ti.transposed(R)
        D[None] = S - ti.transposed(S)

    def V(i, j):
        return i * 2 + j * 7 + int(i == j) * 3

    for i in range(dim):
        for j in range(dim):
            m(i, j)[None] = V(i, j)

    polar()

    tol = 5e-5 if dt == ti.f32 else 1e-12

    for i in range(dim):
        for j in range(dim):
            assert m(i, j)[None] == approx(V(i, j), abs=tol)
            assert I(i, j)[None] == approx(int(i == j), abs=tol)
            assert D(i, j)[None] == approx(0, abs=tol)


def test_polar_decomp():
    for dim in [2, 3]:
        for dt in [ti.f32, ti.f64]:

            @ti.all_archs_with(default_fp=dt)
            def wrapped():
                _test_polar_decomp(dim, dt)

            wrapped()


@ti.all_archs
def test_matrix():
    x = ti.Matrix(2, 2, dt=ti.i32)

    @ti.layout
    def xy():
        ti.root.dense(ti.i, 16).place(x)

    @ti.kernel
    def inc():
        for i in x(0, 0):
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


@ti.all_archs
def _test_mat_inverse_size(n):
    m = ti.Matrix(n, n, dt=ti.f32, shape=())
    M = np.empty(shape=(n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            M[i, j] = i * j + i * 3 + j + 1 + int(i == j) * 4
    assert np.linalg.det(M) != 0

    m.from_numpy(M)

    @ti.kernel
    def invert():
        m[None] = ti.inversed(m[None])

    invert()

    m_np = m.to_numpy()
    np.testing.assert_almost_equal(m_np, np.linalg.inv(M))


def test_mat_inverse():
    for n in range(1, 4):
        _test_mat_inverse_size(n)


@ti.all_archs
def test_unit_vectors():
    a = ti.Vector(3, dt=ti.i32, shape=3)

    @ti.kernel
    def fill():
        for i in ti.static(range(3)):
            a[i] = ti.Vector.unit(3, i)

    fill()

    for i in range(3):
        for j in range(3):
            assert a[i][j] == int(i == j)
