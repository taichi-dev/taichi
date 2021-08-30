import taichi as ti

ti.init(arch=ti.x64)

n = 8

K = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
f = ti.field(ti.f32, shape=n)

@ti.kernel
def fill(A: ti.sparse_matrix_builder(), b: ti.template(),
         interval: ti.i32):
    for i in range(n):
        if i > 0:
            A[i - 1, i] += -1.0
            A[i, i] += 1
        if i < n - 1:
            A[i + 1, i] += -1.0
            A[i, i] += 1.0

        if i % interval == 0:
            b[i] += 1.0


fill(K, f, 3)

A = K.build()
print("A")
print(A)
print("b")
print(f.to_numpy())

print("sparse matrix multiply vector")
x = A @ f.to_numpy()
print(x)

print("sparse matrix solver: Ax = b")
x = A.solve(f.to_numpy())
print(x)

