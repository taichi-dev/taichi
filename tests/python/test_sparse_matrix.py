import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_builder_deprecated_anno(dtype):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_builder(dtype):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_shape(dtype):
    n, m = 8, 9
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             m,
                                             max_num_triplets=100,
                                             dtype=dtype)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, m):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    assert A.shape() == (n, m)


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_element_access(dtype):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            Abuilder[i, i] += i

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        assert A[i, i] == i


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_element_modify(dtype):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            Abuilder[i, i] += i

    fill(Abuilder)
    A = Abuilder.build()
    A[0, 0] = 1024.0
    assert A[0, 0] == 1024.0


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_addition(dtype):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

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


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_subtraction(dtype):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

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


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_scalar_multiplication(dtype):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

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


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_transpose(dtype):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

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


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_elementwise_multiplication(dtype):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

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


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_multiplication(dtype):
    n = 2
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n,
                                             n,
                                             max_num_triplets=100,
                                             dtype=dtype)

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


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_nonsymmetric_multiplication(dtype):
    n, k, m = 2, 3, 4
    Abuilder = ti.linalg.SparseMatrixBuilder(n,
                                             k,
                                             max_num_triplets=100,
                                             dtype=dtype)
    Bbuilder = ti.linalg.SparseMatrixBuilder(k,
                                             m,
                                             max_num_triplets=100,
                                             dtype=dtype)

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
