import numpy as np
import pytest

import taichi as ti


def _eigen_vector_equal(v1, v2, tol):
    if np.linalg.norm(v1) == 0.0:
        assert np.linalg.norm(v2) == 0.0
    else:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        try:
            np.testing.assert_allclose(v1, v2, atol=tol, rtol=tol)
        except AssertionError:
            assert np.allclose(v1, -v2, atol=tol, rtol=tol) or np.allclose(
                v1, 1.j * v2, atol=tol, rtol=tol) or np.allclose(
                    v1, -1.j * v2, atol=tol, rtol=tol)


def _test_eig2x2_real(dt):
    A = ti.Matrix.field(2, 2, dtype=dt, shape=())
    v = ti.Matrix.field(2, 2, dtype=dt, shape=())
    w = ti.Matrix.field(4, 2, dtype=dt, shape=())

    A[None] = [[1, 1], [2, 3]]

    @ti.kernel
    def eigen_solve():
        v[None], w[None] = ti.eig(A[None])

    tol = 1e-5 if dt == ti.f32 else 1e-12
    dtype = np.float32 if dt == ti.f32 else np.float64

    eigen_solve()
    v_np, w_np = np.linalg.eig(A.to_numpy().astype(dtype))
    v_ti = v.to_numpy()[:, 0].astype(dtype)
    w_ti = w.to_numpy()[0::2, :].astype(dtype)

    # sort by eigenvalues
    idx_np = np.argsort(v_np)
    idx_ti = np.argsort(v_ti)

    np.testing.assert_allclose(v_ti[idx_ti], v_np[idx_np], atol=tol, rtol=tol)
    _eigen_vector_equal(w_ti[:, idx_ti[0]], w_np[:, idx_np[0]], tol)
    _eigen_vector_equal(w_ti[:, idx_ti[1]], w_np[:, idx_np[1]], tol)


def _test_eig2x2_complex(dt):
    A = ti.Matrix.field(2, 2, dtype=dt, shape=())
    v = ti.Matrix.field(2, 2, dtype=dt, shape=())
    w = ti.Matrix.field(4, 2, dtype=dt, shape=())

    A[None] = [[1, -1], [1, 1]]

    @ti.kernel
    def eigen_solve():
        v[None], w[None] = ti.eig(A[None])

    tol = 1e-5 if dt == ti.f32 else 1e-12
    dtype = np.float32 if dt == ti.f32 else np.float64

    eigen_solve()
    v_np, w_np = np.linalg.eig(A.to_numpy().astype(dtype))
    v_ti = v.to_numpy().astype(dtype)
    w_ti = w.to_numpy().astype(dtype)
    v_ti_complex = v_ti[:, 0] + v_ti[:, 1] * 1.j
    w_ti_complex = w_ti[0::2, :] + w_ti[1::2, :] * 1.j

    # sort by eigenvalues
    idx_np = np.argsort(v_np)
    idx_ti = np.argsort(v_ti_complex)

    np.testing.assert_allclose(v_ti_complex[idx_ti],
                               v_np[idx_np],
                               atol=tol,
                               rtol=tol)
    _eigen_vector_equal(w_ti_complex[:, idx_ti[0]], w_np[:, idx_np[0]], tol)
    _eigen_vector_equal(w_ti_complex[:, idx_ti[1]], w_np[:, idx_np[1]], tol)


def _test_sym_eig2x2(dt):
    A = ti.Matrix.field(2, 2, dtype=dt, shape=())
    v = ti.Vector.field(2, dtype=dt, shape=())
    w = ti.Matrix.field(2, 2, dtype=dt, shape=())

    A[None] = [[5, 3], [3, 2]]

    @ti.kernel
    def eigen_solve():
        v[None], w[None] = ti.sym_eig(A[None])

    tol = 1e-5 if dt == ti.f32 else 1e-12
    dtype = np.float32 if dt == ti.f32 else np.float64

    eigen_solve()
    v_np, w_np = np.linalg.eig(A.to_numpy().astype(dtype))
    v_ti = v.to_numpy().astype(dtype)
    w_ti = w.to_numpy().astype(dtype)

    # sort by eigenvalues
    idx_np = np.argsort(v_np)
    idx_ti = np.argsort(v_ti)

    np.testing.assert_allclose(v_ti[idx_ti], v_np[idx_np], atol=tol, rtol=tol)
    _eigen_vector_equal(w_ti[:, idx_ti[0]], w_np[:, idx_np[0]], tol)
    _eigen_vector_equal(w_ti[:, idx_ti[1]], w_np[:, idx_np[1]], tol)


def test_eig2x2():
    for func in [_test_eig2x2_real, _test_eig2x2_complex]:
        for fp in [ti.f32, ti.f64]:

            @ti.all_archs_with(default_fp=fp, fast_math=False)
            def wrapped():
                func(fp)

            wrapped()


def test_sym_eig2x2():
    for func in [_test_sym_eig2x2]:
        for fp in [ti.f32, ti.f64]:

            @ti.all_archs_with(default_fp=fp, fast_math=False)
            def wrapped():
                func(fp)

            wrapped()
