import taichi as ti

ti.init(arch=ti.x64)

n = 8

K = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
f = ti.SparseMatrixBuilder(n, 1, max_num_triplets=100)


@ti.kernel
def fill(A: ti.SparseMatrixBuilder, b: ti.SparseMatrixBuilder,
         interval: ti.i32):
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

print(">>>>>>>> Before build: ")
K.print_triplets()

A = K.build()

print(">>>>>>>> After build: ")
A.print()

print(">>>>>>> Summation Test: ")
C = A + A
C.print()

print(">>>> Subtraction Test")
D = A - A
D.print()

print(">>>> Multiplication with scalar")
E = A * 3.0
E.print()

print(">>>> Transpose Test")
F = A.transpose()
F.print()

print(">>>> Matrix Multiplication")
G = E @ A
G.print()

print(">>> Elment-wise Multiplication")
H = E * A
H.print()

print(f">>> Element Access: A[0,0] = {A.get_ele(0,0)}")
