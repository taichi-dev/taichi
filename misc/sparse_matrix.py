import taichi as ti

ti.init(print_preprocessed=True)

n = 8

A = ti.SparseMatrix(n, n, max_num_triplets=1000)

@ti.kernel
def fill(mat: ti.SparseMatrix):
    for i in range(n):
        # mat.insert(i, i, 1.0 + i * i * i)
        mat[i, i] += 1.0 + i * i * i

fill(A)

A.print_triplets()
A.build()
A.print()