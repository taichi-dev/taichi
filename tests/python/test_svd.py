import numpy as np

import taichi as ti
from taichi import approx


@ti.require(ti.extension.data64)
@ti.all_archs_with(fast_math=False)
def test_precision():
    u = ti.field(ti.f64, shape=())
    v = ti.field(ti.f64, shape=())
    w = ti.field(ti.f64, shape=())

    @ti.kernel
    def forward():
        v[None] = ti.sqrt(ti.cast(u[None] + 3.25, ti.f64))
        w[None] = ti.cast(u[None] + 7, ti.f64) / ti.cast(u[None] + 3, ti.f64)

    forward()
    assert v[None]**2 == approx(3.25, abs=1e-12)
    assert w[None] * 3 == approx(7, abs=1e-12)


def mat_equal(A, B, tol=1e-6):
    return np.max(np.abs(A - B)) < tol


def _test_svd(dt, n):
    print(
        f'arch={ti.cfg.arch} default_fp={ti.cfg.default_fp} fast_math={ti.cfg.fast_math} dim={n}'
    )
    A = ti.Matrix.field(n, n, dtype=dt, shape=())
    A_reconstructed = ti.Matrix.field(n, n, dtype=dt, shape=())
    U = ti.Matrix.field(n, n, dtype=dt, shape=())
    UtU = ti.Matrix.field(n, n, dtype=dt, shape=())
    sigma = ti.Matrix.field(n, n, dtype=dt, shape=())
    V = ti.Matrix.field(n, n, dtype=dt, shape=())
    VtV = ti.Matrix.field(n, n, dtype=dt, shape=())

    @ti.kernel
    def run():
        U[None], sigma[None], V[None] = ti.svd(A[None], dt)
        UtU[None] = U[None].transpose() @ U[None]
        VtV[None] = V[None].transpose() @ V[None]
        A_reconstructed[None] = U[None] @ sigma[None] @ V[None].transpose()

    if n == 3:
        A[None] = [[1, 1, 3], [9, -3, 2], [-3, 4, 2]]
    else:
        A[None] = [[1, 1], [2, 3]]

    run()

    tol = 1e-5 if dt == ti.f32 else 1e-12

    assert mat_equal(UtU.to_numpy(), np.eye(n), tol=tol)
    assert mat_equal(VtV.to_numpy(), np.eye(n), tol=tol)
    assert mat_equal(A_reconstructed.to_numpy(), A.to_numpy(), tol=tol)
    for i in range(n):
        for j in range(n):
            if i != j:
                assert sigma[None][i, j] == approx(0)


def test_svd():
    for fp in [ti.f32, ti.f64]:
        for d in [2, 3]:

            @ti.all_archs_with(default_fp=fp, fast_math=False)
            def wrapped():
                _test_svd(fp, d)

            wrapped()


@ti.all_archs
def test_transpose_no_loop():
    A = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
    U = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
    sigma = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
    V = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

    @ti.kernel
    def run():
        U[None], sigma[None], V[None] = ti.svd(A[None])

    run()
    # As long as it passes compilation we are good
