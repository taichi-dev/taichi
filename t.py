import numpy as np

import taichi as ti

ti.init(arch=ti.cuda, debug=True, offline_cache=False)
dtype = ti.f32
n = 10
A = np.random.rand(n, n)
A_psd = np.dot(A, A.transpose())
Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300)

B = np.random.rand(n, n)
B_psd = np.dot(A, A.transpose())
Bbuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300)

b = ti.ndarray(ti.f32, shape=n)


@ti.kernel
def fill(Abuilder: ti.types.sparse_matrix_builder(),
         InputArray: ti.types.ndarray(), b: ti.types.ndarray()):
    for i, j in ti.ndrange(n, n):
        Abuilder[i, j] += InputArray[i, j]
    for i in range(n):
        b[i] = i + 1


fill(Abuilder, A_psd, b)
fill(Bbuilder, B_psd, b)

A = Abuilder.build()
B = Bbuilder.build()
C = A + B
print(type(C))
solver = ti.linalg.SparseSolver(dtype=dtype)
solver.analyze_pattern(C)
solver.factorize(C)
x = solver.solve(b)

C_psd = A_psd + B_psd
res = np.linalg.solve(C_psd, b.to_numpy())

print(x.to_numpy())
print(res)
assert np.allclose(x.to_numpy(), res, rtol=1.0)
