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
def fill_b(b: ti.types.ndarray()):
    b[0] = 1.0
    b[1] = 2.0

@ti.kernel
def print_x(x: ti.types.ndarray()):
    for i in range(2):
        print(x[i])

fill_K(K)
A = K.build(dtype=ti_type)
print(A)

cg = ti.linalg.CG(A, b, x0, max_iter=50, atol=1e-6)

fill_b(b)
cg.solve_cu(x, b)

print_x(x)