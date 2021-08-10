import taichi as ti

ti.init(print_preprocessed=True)

n = 128

A = ti.SparseMatrix(n, n, 1000)

@ti.kernel
def test(mat: ti.SparseMatrix):
    for i in range(3):
        mat.insert(i, i, 1.0 + i * i * i)

test(A)

A.print_triplets()

'''
@ti.kernel
def fill_entries(A: ti.sparse_matrix):
    for i in range(n):
        ti.insert_entry(i, i, 1)

A = ti.sparse_matrix()
fill_entries(A)
A.print_triplets()
'''