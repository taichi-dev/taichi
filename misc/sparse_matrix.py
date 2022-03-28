import taichi as ti

ti.init(arch=ti.x64)

n = 8

K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
f = ti.linalg.SparseMatrixBuilder(n, 1, max_num_triplets=100)


@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(),
         b: ti.types.sparse_matrix_builder(), interval: ti.i32):
    for i in range(n):
        if i > 0:
            A[i - 1, i] += -1.0
            A[i, i] += 1
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

# print(">>>> Summation: C = A + A")
# C = A + A
# print(C)

# print(">>>> Subtraction: D = A - A")
# D = A - A
# print(D)

# print(">>>> Multiplication with a scalar on the right: E = A * 3.0")
# E = A * 3.0
# print(E)

# print(">>>> Multiplication with a scalar on the left: E = 3.0 * A")
# E = 3.0 * A
# print(E)

# print(">>>> Transpose: F = A.transpose()")
# F = A.transpose()
# print(F)

# print(">>>> Matrix multiplication: G = E @ A")
# G = E @ A
# print(G)

# print(">>>> Element-wise multiplication: H = E * A")
# H = E * A
# print(H)

# print(f">>>> Element Access: A[0,0] = {A[0,0]}")
