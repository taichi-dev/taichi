import taichi as ti

ti.init(arch=ti.x64)

n = 8

K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
b = ti.field(ti.f32, shape=n)


@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), b: ti.template(),
         interval: ti.i32):
    for i in range(n):
        A[i, i] += 2.0

        if i % interval == 0:
            b[i] += 1.0


fill(K, b, 3)

A = K.build()
print("A:")
print(A)
print("b:")
print(b.to_numpy())

print("Sparse matrix-vector multiplication (SpMV): A @ b =")
x = A @ b
print(x)

print("Solving sparse linear systems Ax = b with the solution x:")
solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(A)
solver.factorize(A)
x = solver.solve(b)
print(x)
isSuccess = solver.info()
print(f"Computation was successful?: {isSuccess}")
