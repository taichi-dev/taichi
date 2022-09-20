import numpy as np
from scipy.sparse import coo_matrix

import taichi as ti

ti.init(arch=ti.cuda)


@ti.kernel
def init_b(b: ti.types.ndarray(), nrows: ti.i32):
    for i in range(nrows):
        b[i] = 1.0 + i / nrows


@ti.kernel
def print_x(x: ti.types.ndarray(), ncols: ti.i32):
    for i in range(ncols):
        print(x[i], end=' ')
    print()


"""
Generate a positive definite matrix with a given number of rows and columns.
Reference: https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices
"""
matrixSize = 10
A = np.random.rand(matrixSize, matrixSize)
A_psd = np.dot(A, A.transpose())

A_raw_coo = coo_matrix(A_psd)
nrows, ncols = A_raw_coo.shape
nnz = A_raw_coo.nnz

A_csr = A_raw_coo.tocsr()
b = ti.ndarray(shape=nrows, dtype=ti.f32)
init_b(b, nrows)

print(">> solve Ax = b using Cusolver ......... ")
A_coo = A_csr.tocoo()
d_row_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
d_col_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
d_val_coo = ti.ndarray(shape=nnz, dtype=ti.f32)
d_row_coo.from_numpy(A_coo.row)
d_col_coo.from_numpy(A_coo.col)
d_val_coo.from_numpy(A_coo.data)

A_ti = ti.linalg.SparseMatrix(n=nrows, m=ncols, dtype=ti.float32)
A_ti.build_coo(d_row_coo, d_col_coo, d_val_coo)
x_ti = ti.ndarray(shape=ncols, dtype=ti.float32)
solver = ti.linalg.SparseSolver()
solver.solve_cu(A_ti, b, x_ti)
ti.sync()
print_x(x_ti, ncols)
ti.sync()

print(">> solve Ax = b using Numpy ......... ")
b_np = b.to_numpy()
x_np = np.linalg.solve(A_psd, b_np)
print(x_np)

print(
    f"The solution is identical?: {np.allclose(x_ti.to_numpy(), x_np, atol=1e-1)}"
)
