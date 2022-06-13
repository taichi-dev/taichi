import taichi as ti

ti.init(arch=ti.x64)

n = 8

K = ti.linalg.SparseMatrixBuilder(n,
                                  n,
                                  max_num_triplets=100,
                                  dtype=ti.f32,
                                  storage_format='col_major')
f = ti.linalg.SparseMatrixBuilder(n,
                                  1,
                                  max_num_triplets=100,
                                  dtype=ti.f32,
                                  storage_format='col_major')


@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(),
         b: ti.types.sparse_matrix_builder(), interval: ti.i32):
    for i in range(n):
        if i > 0:
            A[i - 1, i] += -2.0
            A[i, i] += 1.0
        if i < n - 1:
            A[i + 1, i] += -1.0
            A[i, i] += 1.0

        if i % interval == 0:
            b[i, 0] += 1.0


fill(K, f, 3)

print(">>>> K.print_triplets()")
K.print_triplets()

A = K.build()

print(">>>> A = K.build()")
print(A)

print(">>>> Summation: B = A + A")
B = A + A
print(B)

print(">>>> Summation: B += A")
B += A
print(B)

print(">>>> Subtraction: C = B - A")
C = B - A
print(C)

print(">>>> Subtraction: C -= A")
C -= A
print(C)

print(">>>> Multiplication with a scalar on the right: D = A * 3.0")
D = A * 3.0
print(D)

print(">>>> Multiplication with a scalar on the left: D = 3.0 * A")
D = 3.0 * A
print(D)

print(">>>> Transpose: E = D.transpose()")
E = D.transpose()
print(E)

print(">>>> Matrix multiplication: F= E @ A")
F = E @ A
print(F)

print(f">>>> Element Access: F[0,0] = {F[0,0]}")
