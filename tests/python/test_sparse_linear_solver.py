import numpy as np
import pytest

import taichi as ti
from tests import test_utils

"""
The symmetric positive definite matrix is created in matlab using the following script:
    A = diag([1,2,3,4]);
    OrthM = [1 0 1 0; -1 -2 0 1; 0 1 -1 0; 0, 1, 0 1];
    U = orth(OrthM);
    Aarray = U * A * U';
    b = [1,2,3,4]';
    res = inv(A) * b;
""" # yapf: disable

Aarray = np.array([[
    2.73999501130921, 0.518002544441220, 0.745119303009342, 0.0508907745638859
], [0.518002544441220, 1.45111665837647, 0.757997555750432, 0.290885785873098],
                   [
                       0.745119303009342, 0.757997555750432, 2.96711176987733,
                       -0.518002544441220
                   ],
                   [
                       0.0508907745638859, 0.290885785873098,
                       -0.518002544441220, 2.84177656043698
                   ]])

res = np.array([
    -0.0754984396447588, 0.469972700892492, 1.18527357933586, 1.57686870529319
])


@pytest.mark.parametrize("dtype", [ti.f32])
@pytest.mark.parametrize("solver_type", ["LLT", "LDLT", "LU"])
@pytest.mark.parametrize("ordering", ["AMD", "COLAMD"])
@test_utils.test(arch=ti.cpu)
def test_sparse_LLT_solver(dtype, solver_type, ordering):
    n = 4
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
    b = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(),
             InputArray: ti.types.ndarray(), b: ti.template()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, Aarray, b)
    A = Abuilder.build()
    solver = ti.linalg.SparseSolver(dtype=dtype,
                                    solver_type=solver_type,
                                    ordering=ordering)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)
    for i in range(n):
        assert x[i] == test_utils.approx(res[i])


@test_utils.test(arch=ti.cuda)
def test_gpu_sparse_solver():
    from scipy.sparse import coo_matrix

    @ti.kernel
    def init_b(b: ti.types.ndarray(), nrows: ti.i32):
        for i in range(nrows):
            b[i] = 1.0 + i / nrows

    """
    Generate a positive definite matrix with a given number of rows and columns.
    Reference: https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices
    """
    matrixSize = 10
    A = np.random.rand(matrixSize, matrixSize)
    A_psd = np.dot(A, A.transpose())

    A_raw_coo = coo_matrix(A_psd)
    nrows, ncols = A_raw_coo.shape
    nnz = A_raw_coo.nnz

    A_csr = A_raw_coo.tocsr()
    b = ti.ndarray(shape=nrows, dtype=ti.f32)
    init_b(b, nrows)

    # solve Ax = b using cusolver
    A_coo = A_csr.tocoo()
    d_row_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
    d_col_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
    d_val_coo = ti.ndarray(shape=nnz, dtype=ti.f32)
    d_row_coo.from_numpy(A_coo.row)
    d_col_coo.from_numpy(A_coo.col)
    d_val_coo.from_numpy(A_coo.data)

    A_ti = ti.linalg.SparseMatrix(n=nrows, m=ncols, dtype=ti.float32)
    A_ti.build_coo(d_row_coo, d_col_coo, d_val_coo)
    x_ti = ti.ndarray(shape=ncols, dtype=ti.float32)
    solver = ti.linalg.SparseSolver()
    solver.solve_cu(A_ti, b, x_ti)
    ti.sync()

    # solve Ax=b using numpy
    b_np = b.to_numpy()
    x_np = np.linalg.solve(A_psd, b_np)
    assert (np.allclose(x_ti.to_numpy(), x_np, atol=1e-1))

    # solve Ax=b using cusolver refectorization
    solver = ti.linalg.SparseSolver()
    solver.analyze_pattern(A_ti)
    solver.factorize(A_ti)
    solver.solve_rf(A_ti, b, x_ti)
    ti.sync()
    assert (np.allclose(x_ti.to_numpy(), x_np, atol=1e-1))
