import pytest

import taichi as ti
from tests import test_utils

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_builder_deprecated_anno(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_builder(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_build_sparse_matrix_frome_ndarray(dtype, storage_format):
    n = 8
    triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=n)
    A = ti.linalg.SparseMatrix(n=10,
                               m=10,
                               dtype=ti.f32,
                               storage_format=storage_format)

    @ti.kernel
    def fill(triplets: ti.types.ndarray()):
        for i in range(n):
            triplet = ti.Vector([i, i, i], dt=ti.f32)
            triplets[i] = triplet

    fill(triplets)
    A.build_from_ndarray(triplets)

    for i in range(n):
        assert A[i, i] == i

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_shape(dtype, storage_format):
    n, m = 8, 9
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             m,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, m):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    assert A.shape == (n, m)

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_element_access(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            Abuilder[i, i] += i

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        assert A[i, i] == i

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_element_modify(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            Abuilder[i, i] += i

    fill(Abuilder)
    A = Abuilder.build()
    A[0, 0] = 1024.0
    assert A[0, 0] == 1024.0

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_addition(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(),
             Bbuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A + B
    for i in range(n):
        for j in range(n):
            assert C[i, j] == 2 * i

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_subtraction(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(),
             Bbuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A - B
    for i in range(n):
        for j in range(n):
            assert C[i, j] == 2 * j

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_scalar_multiplication(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    B = A * 3.0
    for i in range(n):
        for j in range(n):
            assert B[i, j] == 3 * (i + j)

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_transpose(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    B = A.transpose()
    for i in range(n):
        for j in range(n):
            assert B[i, j] == A[j, i]

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_elementwise_multiplication(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(),
             Bbuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A * B
    for i in range(n):
        for j in range(n):
            assert C[i, j] == (i + j) * (i - j)

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_multiplication(dtype, storage_format):
    n = 2
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(),
             Bbuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A @ B
    assert C[0, 0] == 1.0
    assert C[0, 1] == 0.0
    assert C[1, 0] == 2.0
    assert C[1, 1] == -1.0

@pytest.mark.parametrize('dtype, storage_format', [(ti.f32, 'col_major'),
                                                   (ti.f32, 'row_major'),
                                                   (ti.f64, 'col_major'),
                                                   (ti.f64, 'row_major')])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_nonsymmetric_multiplication(dtype, storage_format):
    n, k, m = 2, 3, 4
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             k,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(k,
                                             m,
                                             max_num_triplets=100,
                                             dtype=dtype,
                                             storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(),
             Bbuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, k):
            Abuilder[i, j] += i + j
        for i, j in ti.ndrange(k, m):
            Bbuilder[i, j] -= i + j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A @ B
    GT = [[-5, -8, -11, -14], [-8, -14, -20, -26]]
    for i in range(n):
        for j in range(m):
            assert C[i, j] == GT[i][j]

@test_utils.test(arch=ti.cuda)
def test_gpu_sparse_matrix():
    import numpy as np
    h_coo_row = np.asarray([1, 0, 0, 0, 2, 2, 2, 3, 3], dtype=np.int32)
    h_coo_col = np.asarray([1, 0, 2, 3, 0, 2, 3, 1, 3], dtype=np.int32)
    h_coo_val = np.asarray([4.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                           dtype=np.float32)
    h_X = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    h_Y = np.asarray([19.0, 8.0, 51.0, 52.0], dtype=np.float32)

    # Data structure for building the CSR matrix A using Taichi Sparse Matrix
    idx_dt = ti.int32
    val_dt = ti.f32
    d_coo_row = ti.ndarray(shape=9, dtype=idx_dt)
    d_coo_col = ti.ndarray(shape=9, dtype=idx_dt)
    d_coo_val = ti.ndarray(shape=9, dtype=val_dt)
    # Dense vector x
    X = ti.ndarray(shape=4, dtype=val_dt)
    # Results for A @ x
    Y = ti.ndarray(shape=4, dtype=val_dt)

    # Initialize the CSR matrix and vectors with numpy array
    d_coo_row.from_numpy(h_coo_row)
    d_coo_col.from_numpy(h_coo_col)
    d_coo_val.from_numpy(h_coo_val)
    X.from_numpy(h_X)
    Y.fill(0.0)

    # Define the CSR matrix A
    A = ti.linalg.SparseMatrix(n=4, m=4, dtype=ti.f32)

    # Build the CSR matrix A with Taichi ndarray
    A.build_coo(d_coo_row, d_coo_col, d_coo_val)

    # Compute Y = A @ X
    A.spmv(X, Y)
    for i in range(4):
        assert Y[i] == h_Y[i]


@pytest.mark.parametrize('N', [5])
@test_utils.test(arch=ti.cuda)
def test_gpu_sparse_matrix_ops(N):
    import numpy as np
    from numpy.random import default_rng
    from scipy import stats
    from scipy.sparse import coo_matrix, random

    seed = 2
    np.random.seed(seed)
    rng = default_rng(seed)
    rvs = stats.poisson(3, loc=1).rvs
    np_dtype = np.float32
    idx_dt = ti.int32
    val_dt = ti.float32

    n_rows = N
    n_cols = N - 1

    S1 = random(n_rows, n_cols, density=0.5, random_state=rng,
                data_rvs=rvs).astype(np_dtype).tocoo()
    S2 = random(n_rows, n_cols, density=0.5, random_state=rng,
                data_rvs=rvs).astype(np_dtype).tocoo()

    nnz_A = len(S1.data)
    nnz_B = len(S2.data)

    row_coo_A = ti.ndarray(shape=nnz_A, dtype=idx_dt)
    col_coo_A = ti.ndarray(shape=nnz_A, dtype=idx_dt)
    value_coo_A = ti.ndarray(shape=nnz_A, dtype=val_dt)
    row_coo_B = ti.ndarray(shape=nnz_B, dtype=idx_dt)
    col_coo_B = ti.ndarray(shape=nnz_B, dtype=idx_dt)
    value_coo_B = ti.ndarray(shape=nnz_B, dtype=val_dt)

    row_coo_A.from_numpy(S1.row)
    col_coo_A.from_numpy(S1.col)
    value_coo_A.from_numpy(S1.data)

    row_coo_B.from_numpy(S2.row)
    col_coo_B.from_numpy(S2.col)
    value_coo_B.from_numpy(S2.data)

    A = ti.linalg.SparseMatrix(n=n_rows, m=n_cols, dtype=ti.f32)
    B = ti.linalg.SparseMatrix(n=n_rows, m=n_cols, dtype=ti.f32)
    A.build_coo(row_coo_A, col_coo_A, value_coo_A)
    B.build_coo(row_coo_B, col_coo_B, value_coo_B)

    def verify(scipy_spm, taichi_spm):
        scipy_spm = scipy_spm.tocoo()
        for i, j, v in zip(scipy_spm.row, scipy_spm.col, scipy_spm.data):
            # assert v == taichi_spm[i, j]
            assert v == test_utils.approx(taichi_spm[i, j], rel=1e-5)

    C = A + B
    S3 = S1 + S2
    verify(S3, C)

    D = C - A
    S4 = S3 - S1
    verify(S4, D)

    E = A * 2.5
    S5 = S1 * 2.5
    verify(S5, E)
    
    F = A * 2.5
    S6 = S1 * 2.5
    verify(S6, F)

    G = A.transpose()
    S7 = S1.T
    verify(S7, G)

    H = A @ B.transpose()
    S8 = S1 @ S2.T
    verify(S8, H)
