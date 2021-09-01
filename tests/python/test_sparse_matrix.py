import taichi as ti


@ti.test(arch=ti.cpu)
def test_sparse_matrix_builder():
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j


@ti.test(arch=ti.cpu)
def test_sparse_matrix_element_access():
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.sparse_matrix_builder()):
        for i in range(n):
            Abuilder[i, i] += i

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        assert A[i, i] == i


@ti.test(arch=ti.cpu)
def test_sparse_matrix_addition():
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
    Bbuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.sparse_matrix_builder(),
             Bbuilder: ti.sparse_matrix_builder()):
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


@ti.test(arch=ti.cpu)
def test_sparse_matrix_subtraction():
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
    Bbuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.sparse_matrix_builder(),
             Bbuilder: ti.sparse_matrix_builder()):
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


@ti.test(arch=ti.cpu)
def test_sparse_matrix_scalar_multiplication():
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    B = A * 3.0
    for i in range(n):
        for j in range(n):
            assert B[i, j] == 3 * (i + j)


@ti.test(arch=ti.cpu)
def test_sparse_matrix_transpose():
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    B = A.transpose()
    for i in range(n):
        for j in range(n):
            assert B[i, j] == A[j, i]


@ti.test(arch=ti.cpu)
def test_sparse_matrix_elementwise_multiplication():
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
    Bbuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.sparse_matrix_builder(),
             Bbuilder: ti.sparse_matrix_builder()):
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


@ti.test(arch=ti.cpu)
def test_sparse_matrix_multiplication():
    n = 2
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
    Bbuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.sparse_matrix_builder(),
             Bbuilder: ti.sparse_matrix_builder()):
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


@ti.test(arch=ti.cpu)
def test_sparse_matrix_nonsymmetric_multiplication():
    n, k, m = 2, 3, 4
    Abuilder = ti.SparseMatrixBuilder(n, k, max_num_triplets=100)
    Bbuilder = ti.SparseMatrixBuilder(k, m, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.sparse_matrix_builder(),
             Bbuilder: ti.sparse_matrix_builder()):
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
