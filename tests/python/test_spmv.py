import taichi as ti


@ti.test(arch=ti.cpu)
def test_sparse_matrix_vector_multiplication1():
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
    b = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.linalg.sparse_matrix_builder(), b: ti.template()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i

        for i in range(n):
            b[i] = 1.0

    fill(Abuilder, b)
    A = Abuilder.build()
    x = A @ b
    for i in range(n):
        assert x[i] == 8 * i


@ti.test(arch=ti.cpu)
def test_sparse_matrix_vector_multiplication2():
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
    b = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.linalg.sparse_matrix_builder(), b: ti.template()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i - j

        for i in range(n):
            b[i] = 1.0

    fill(Abuilder, b)
    A = Abuilder.build()

    x = A @ b
    import numpy as np
    res = np.array([-28, -20, -12, -4, 4, 12, 20, 28])
    for i in range(n):
        assert x[i] == res[i]


@ti.test(arch=ti.cpu)
def test_sparse_matrix_vector_multiplication3():
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
    b = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.linalg.sparse_matrix_builder(), b: ti.template()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

        for i in range(n):
            b[i] = 1.0

    fill(Abuilder, b)
    A = Abuilder.build()

    x = A @ b
    import numpy as np
    res = np.array([28, 36, 44, 52, 60, 68, 76, 84])
    for i in range(n):
        assert x[i] == res[i]
