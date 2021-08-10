import taichi as ti

ti.init(arch=ti.x64)

n = 8

K = ti.SparseMatrix(n, n, max_num_triplets=1000)
f = ti.SparseMatrix(n, 1, max_num_triplets=1000)


@ti.kernel
def fill(A: ti.SparseMatrix, b: ti.SparseMatrix, interval: ti.i32):
    for i in range(n):
        if i > 0:
            A[i - 1, i] += -1.0
            A[i, i] += 1.0
        if i < n - 1:
            A[i + 1, i] += -1.0
            A[i, i] += 1.0

        if i % interval == 0:
            b[i, 0] += 1.0


fill(K, f, 3)

print()

K.print_triplets()
K.build()

print('K = ')
K.print()
print()

print('f = ')
f.build()
f.print()

print()
print('u = ')
K.solve(f)
# TODO: where to store the results?
