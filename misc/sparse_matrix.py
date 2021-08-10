import taichi as ti

ti.init(print_preprocessed=True)

n = 8

A = ti.SparseMatrix(n, n, max_num_triplets=1000)
# b = ti.SparseMatrix(n, 1, max_num_triplets=1000)

@ti.kernel
def fill(mat: ti.SparseMatrix):#, interval: ti.i32):
    for i in range(n):
        if i > 0:
            mat[i - 1, i] += -1.0
            mat[i, i] += 1.0
        if i < n - 1:
            mat[i + 1, i] += -1.0
            mat[i, i] += 1.0

        # if i % interval == 0:
        #    b[i, 0] += 1.0


fill(A)

A.print_triplets()
A.build()
A.print()

exit()

b.build()
b.print()