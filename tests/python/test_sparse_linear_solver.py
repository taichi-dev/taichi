import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@pytest.mark.parametrize("solver_type", ["LLT", "LDLT", "LU"])
@pytest.mark.parametrize("ordering", ["AMD", "COLAMD"])
@test_utils.test(arch=ti.x64)
def test_sparse_LLT_solver(dtype, solver_type, ordering):
    n = 10
    A = np.random.rand(n, n)
    A_psd = np.dot(A, A.transpose())
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)
    b = ti.field(dtype=dtype, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(),
             InputArray: ti.types.ndarray(), b: ti.template()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build()
    solver = ti.linalg.SparseSolver(dtype=dtype,
                                    solver_type=solver_type,
                                    ordering=ordering)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)

    res = np.linalg.solve(A_psd, b.to_numpy())
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)


@pytest.mark.parametrize("dtype", [ti.f32])
@pytest.mark.parametrize("solver_type", ["LLT", "LDLT", "LU"])
@pytest.mark.parametrize("ordering", ["AMD", "COLAMD"])
@test_utils.test(arch=ti.cpu)
def test_sparse_solver_ndarray_vector(dtype, solver_type, ordering):
    n = 10
    A = np.random.rand(n, n)
    A_psd = np.dot(A, A.transpose())
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300)
    b = ti.ndarray(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(),
             InputArray: ti.types.ndarray(), b: ti.types.ndarray()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build()
    solver = ti.linalg.SparseSolver(dtype=dtype,
                                    solver_type=solver_type,
                                    ordering=ordering)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)

    res = np.linalg.solve(A_psd, b.to_numpy())
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)


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
    A_builder = ti.linalg.SparseMatrixBuilder(num_rows=nrows,
                                              num_cols=ncols,
                                              dtype=ti.f32,
                                              max_num_triplets=nnz)

    @ti.kernel
    def fill(A_builder: ti.types.sparse_matrix_builder(),
             row_coo: ti.types.ndarray(), col_coo: ti.types.ndarray(),
             val_coo: ti.types.ndarray()):
        for i in range(nnz):
            A_builder[row_coo[i], col_coo[i]] += val_coo[i]

    fill(A_builder, A_coo.row, A_coo.col, A_coo.data)
    A_ti = A_builder.build()
    x_ti = ti.ndarray(shape=ncols, dtype=ti.float32)

    # solve Ax=b using numpy
    b_np = b.to_numpy()
    x_np = np.linalg.solve(A_psd, b_np)

    # solve Ax=b using cusolver refectorization
    solver = ti.linalg.SparseSolver()
    solver.analyze_pattern(A_ti)
    solver.factorize(A_ti)
    x_ti = solver.solve(b)
    ti.sync()
    assert (np.allclose(x_ti.to_numpy(), x_np, rtol=5.0e-3))

    # solve Ax = b using compute function
    solver = ti.linalg.SparseSolver()
    solver.compute(A_ti)
    x_cti = solver.solve(b)
    ti.sync()
    assert (np.allclose(x_cti.to_numpy(), x_np, rtol=5.0e-3))


@pytest.mark.parametrize("dtype", [ti.f32])
@pytest.mark.parametrize("solver_type", ["LLT", "LU"])
@test_utils.test(arch=ti.cuda)
def test_gpu_sparse_solver2(dtype, solver_type):
    n = 10
    A = np.random.rand(n, n)
    A_psd = np.dot(A, A.transpose())
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300)
    b = ti.ndarray(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(),
             InputArray: ti.types.ndarray(), b: ti.types.ndarray()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build()
    solver = ti.linalg.SparseSolver(dtype=dtype, solver_type=solver_type)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)

    res = np.linalg.solve(A_psd, b.to_numpy())
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)
