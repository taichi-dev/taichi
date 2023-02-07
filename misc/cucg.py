import numpy as np

import taichi as ti

ti.init(arch=ti.cuda)

ti_type = ti.f32
np_type = np.float32
K = ti.linalg.SparseMatrixBuilder(2, 2, max_num_triplets=50, dtype=ti_type)
b = ti.ndarray(dtype=ti_type, shape=2)
x0 = ti.ndarray(dtype=ti_type, shape=2)
x = ti.ndarray(dtype=ti_type, shape=2)

@ti.kernel
def fill_K(K: ti.types.sparse_matrix_builder()):
    K[0, 0] += 4.0
    K[0, 1] += 1.0
    K[1, 0] += 1.0
    K[1, 1] += 3.0

@ti.kernel
def print_x(x: ti.types.ndarray()):
    for i in range(2):
        print(x[i])

fill_K(K)
A = K.build(dtype=ti_type)
print(A)

np_b = np.asarray([1.0, 2.0], dtype=np_type)
b.from_numpy(np_b)
print("b =", b.to_numpy())

cg = ti.linalg.CG(A, b, x0, max_iter=50, atol=1e-6)
x, status_code = cg.solve()

print_x(x)