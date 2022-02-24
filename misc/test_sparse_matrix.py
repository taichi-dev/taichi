import taichi as ti

ti.init(arch=ti.x64)

n = 8

K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=ti.f32)


@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), interval: ti.i32):
    for i in range(n):
        if i > 0:
            A[i - 1, i] += -1.0
            A[i, i] += 1
        if i < n - 1:
            A[i + 1, i] += -1.0
            A[i, i] += 1.0

fill(K, 3)

print(">>>> K.print_triplets()")
K.print_triplets()