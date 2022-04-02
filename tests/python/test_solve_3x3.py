import numpy as np
import pytest

import taichi as ti
from tests import test_utils


def _solve_vector_equal(v1, v2, tol):
    if np.linalg.norm(v1) == 0.0:
        assert np.linalg.norm(v2) == 0.0
    else:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        np.testing.assert_allclose(v1, v2, atol=tol, rtol=tol)


def _test_solve_2x2(dt, a00):
    A = ti.Matrix.field(2, 2, dtype=dt, shape=())
    b = ti.Vector.field(2, dtype=dt, shape=())
    x = ti.Vector.field(2, dtype=dt, shape=())

    @ti.kernel
    def solve_2x2():
        A[None] = ti.Matrix([[a00, 2.0], [2.0, 3.0]])
        b[None] = ti.Vector([3.0, 15.0])
        x[None] = ti.gsolve(A[None], b[None])

    solve_2x2()

    tol = 1e-5 if dt == ti.f32 else 1e-12
    dtype = np.float32 if dt == ti.f32 else np.float64
    x_np = np.linalg.solve(A[None].to_numpy().astype(dtype),
                           b[None].to_numpy().astype(dtype))
    x_ti = x[None].to_numpy().astype(dtype)

    idx_np = np.argsort(x_np)
    idx_ti = np.argsort(x_ti)
    np.testing.assert_allclose(x_np[idx_np], x_ti[idx_ti], atol=tol, rtol=tol)
    _solve_vector_equal(x_ti, x_np, tol)


def _test_solve_3x3(dt, a00):
    A = ti.Matrix.field(3, 3, dtype=dt, shape=())
    b = ti.Vector.field(3, dtype=dt, shape=())
    x = ti.Vector.field(3, dtype=dt, shape=())

    @ti.kernel
    def solve_3x3():
        A[None] = ti.Matrix([[a00, 2.0, -4.0], [2.0, 3.0, 3.0], [5.0, -3,
                                                                 1.0]])
        b[None] = ti.Vector([3.0, 15.0, 14.0])
        x[None] = ti.ge_solve(A[None], b[None])

    solve_3x3()

    tol = 1e-5 if dt == ti.f32 else 1e-12
    dtype = np.float32 if dt == ti.f32 else np.float64
    x_np = np.linalg.solve(A[None].to_numpy().astype(dtype),
                           b[None].to_numpy().astype(dtype))
    x_ti = x[None].to_numpy().astype(dtype)

    idx_np = np.argsort(x_np)
    idx_ti = np.argsort(x_ti)
    np.testing.assert_allclose(x_np[idx_np], x_ti[idx_ti], atol=tol, rtol=tol)
    _solve_vector_equal(x_ti, x_np, tol)


# @pytest.mark.parametrize('a00', [i for i in range(10)])
# @test_utils.test(default_fp=ti.f32, fast_math=False)
# def test_solve_2x2_f32(a00):
#     _test_solve_2x2(ti.f32, a00)

# @pytest.mark.parametrize('a00', [i for i in range(10)])
# @test_utils.test(require=ti.extension.data64,
#                  default_fp=ti.f64,
#                  fast_math=False)
# def test_solve_2x2_f64(a00):
#     _test_solve_2x2(ti.f64, a00)


@pytest.mark.parametrize('a00', [i for i in range(10)])
@test_utils.test(default_fp=ti.f32, fast_math=False)
def test_solve_3x3_f32(a00):
    _test_solve_3x3(ti.f32, a00)


@pytest.mark.parametrize('a00', [i for i in range(10)])
@test_utils.test(require=ti.extension.data64,
                 default_fp=ti.f64,
                 fast_math=False)
def test_solve_3x3_f64(a00):
    _test_solve_3x3(ti.f64, a00)
