import taichi as ti


@ti.test(ti.cpu)
def test_sparse_matrix_builder():
    ti.init(arch=ti.cpu)
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.SparseMatrixBuilder):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += float(i + j)

    fill(Abuilder)
    A = ti.SparseMatrix(n, n)
    Abuilder.build(A)
    for i in range(n):
        for j in range(n):
            assert A.get_ele(i, j) == float(i + j)


@ti.test(ti.cpu)
def test_sparse_matrix_element_access():
    ti.init(arch=ti.cpu)
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.SparseMatrixBuilder):
        for i in range(n):
            Abuilder[i, i] += 1.0

    fill(Abuilder)
    A = ti.SparseMatrix(n, n)
    Abuilder.build(A)
    for i in range(n):
        assert A.get_ele(i, i) == 1.0


@ti.test(ti.cpu)
def test_sparse_matrix_addition():
    ti.init(arch=ti.cpu)
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
    Bbuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.SparseMatrixBuilder,
             Bbuilder: ti.SparseMatrixBuilder):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += float(i + j)
            Bbuilder[i, j] += float(i - j)

    fill(Abuilder, Bbuilder)
    A = ti.SparseMatrix(n, n)
    B = ti.SparseMatrix(n, n)
    Abuilder.build(A)
    Bbuilder.build(B)
    C = A + B
    for i in range(n):
        for j in range(n):
            assert C.get_ele(i, j) == float(2.0 * i)


@ti.test(ti.cpu)
def test_sparse_matrix_subtraction():
    ti.init(arch=ti.cpu)
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
    Bbuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.SparseMatrixBuilder,
             Bbuilder: ti.SparseMatrixBuilder):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += float(i + j)
            Bbuilder[i, j] += float(i - j)

    fill(Abuilder, Bbuilder)
    A = ti.SparseMatrix(n, n)
    B = ti.SparseMatrix(n, n)
    Abuilder.build(A)
    Bbuilder.build(B)
    C = A - B
    for i in range(n):
        for j in range(n):
            assert C.get_ele(i, j) == float(2.0 * j)


@ti.test(ti.cpu)
def test_sparse_matrix_scalar_multiplication():
    ti.init(arch=ti.cpu)
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.SparseMatrixBuilder):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += float(i + j)

    fill(Abuilder)
    A = ti.SparseMatrix(n, n)
    Abuilder.build(A)
    B = A * 3.0
    for i in range(n):
        for j in range(n):
            assert B.get_ele(i, j) == float(i + j) * 3.0


@ti.test(ti.cpu)
def test_sparse_matrix_transpose():
    ti.init(arch=ti.cpu)
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.SparseMatrixBuilder):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += float(i + j)

    fill(Abuilder)
    A = ti.SparseMatrix(n, n)
    Abuilder.build(A)
    B = A.transpose()
    for i in range(n):
        for j in range(n):
            assert B.get_ele(i, j) == A.get_ele(j, i)


@ti.test(ti.cpu)
def test_sparse_matrix_elementwise_multiplication():
    ti.init(arch=ti.cpu)
    n = 8
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
    Bbuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.SparseMatrixBuilder,
             Bbuilder: ti.SparseMatrixBuilder):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += float(i + j)
            Bbuilder[i, j] += float(i - j)

    fill(Abuilder, Bbuilder)
    A = ti.SparseMatrix(n, n)
    B = ti.SparseMatrix(n, n)
    Abuilder.build(A)
    Bbuilder.build(B)
    C = A * B
    for i in range(n):
        for j in range(n):
            assert C.get_ele(i, j) == float(i + j) * float(i - j)


@ti.test(ti.cpu)
def test_sparse_matrix_multiplication():
    ti.init(arch=ti.cpu)
    n = 2
    Abuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
    Bbuilder = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.SparseMatrixBuilder,
             Bbuilder: ti.SparseMatrixBuilder):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += float(i + j)
            Bbuilder[i, j] += float(i - j)

    fill(Abuilder, Bbuilder)
    A = ti.SparseMatrix(n, n)
    B = ti.SparseMatrix(n, n)
    Abuilder.build(A)
    Bbuilder.build(B)
    C = A @ B
    assert C.get_ele(0, 0) == 1.0
    assert C.get_ele(0, 1) == 0.0
    assert C.get_ele(1, 0) == 2.0
    assert C.get_ele(1, 1) == -1.0
