import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize("ti_dtype", [ti.f32, ti.f64])
@test_utils.test(arch=[ti.cpu])
def test_cg(ti_dtype):
    n = 10
    A = np.random.rand(n, n)
    A_psd = np.dot(A, A.transpose())
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=ti_dtype)
    b = ti.ndarray(dtype=ti_dtype, shape=n)
    x0 = ti.ndarray(dtype=ti_dtype, shape=n)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        InputArray: ti.types.ndarray(),
        b: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build(dtype=ti_dtype)
    cg = ti.linalg.SparseCG(A, b, x0, max_iter=50, atol=1e-6)
    x, exit_code = cg.solve()
    res = np.linalg.solve(A_psd, b.to_numpy())
    assert exit_code == True
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)


@pytest.mark.parametrize("ti_dtype", [ti.f32])
@test_utils.test(arch=[ti.cuda])
def test_cg_cuda(ti_dtype):
    n = 10
    A = np.random.rand(n, n)
    A_psd = np.dot(A, A.transpose())
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=ti_dtype)
    b = ti.ndarray(dtype=ti_dtype, shape=n)
    x0 = ti.ndarray(dtype=ti_dtype, shape=n)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        InputArray: ti.types.ndarray(),
        b: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build(dtype=ti_dtype)
    cg = ti.linalg.SparseCG(A, b, x0, max_iter=50, atol=1e-6)
    x, exit_code = cg.solve()
    res = np.linalg.solve(A_psd, b.to_numpy())
    assert exit_code == True
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)
