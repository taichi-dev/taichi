import numpy as np

import taichi as ti

ti.init(arch=ti.x64, offline_cache=False)

K = ti.linalg.SparseMatrixBuilder(2, 2, max_num_triplets=50)


@ti.kernel
def fill_K(K: ti.types.sparse_matrix_builder()):
    K[0, 0] += 4.0
    K[0, 1] += 1.0
    K[1, 0] += 1.0
    K[1, 1] += 3.0


fill_K(K)
A = K.build()

b = np.array([1.0, 2.0], dtype=np.float32)
x0 = np.array([0.0, 0.0], dtype=np.float32)
cg = ti.linalg.CG(A, b, x0, max_iter=50, atol=1e-6)

x, exit_code = cg.solve()

print(x, exit_code)
