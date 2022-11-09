import taichi as ti

ti.init(arch=ti.cuda, debug=True)

n = 2

K = ti.linalg.SparseMatrixBuilder(n,
                                  n,
                                  max_num_triplets=100,
                                  dtype=ti.f32,
                                  storage_format='col_major')

# K.test_ndarray()


@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    for i in range(n):
        A[i, i] += 2


fill(K)

K.print_ndarray_data()
