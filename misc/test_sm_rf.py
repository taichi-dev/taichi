import numpy as np
import scipy.io as sio

import taichi as ti

ti.init(arch=ti.cuda)


@ti.kernel
def print_x(x: ti.types.ndarray(), ncols: ti.i32, length: ti.i32):
    for i in range(ncols - length, ncols):
        print(x[i])


print(">> load sparse matrix........")
A_raw_coo = sio.mmread('misc/lap2D_5pt_n100.mtx')
A_csr = A_raw_coo.tocsr()
A_coo = A_csr.tocoo()

nrows, ncols = A_coo.shape
nnz = A_coo.nnz
ti.sync()
print(">> solve Ax = b using CuSparseSolver ......... ")
d_row_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
d_col_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
d_val_coo = ti.ndarray(shape=nnz, dtype=ti.f32)
d_row_coo.from_numpy(A_coo.row)
d_col_coo.from_numpy(A_coo.col)
d_val_coo.from_numpy(A_coo.data)

A_ti = ti.linalg.SparseMatrix(n=nrows, m=ncols, dtype=ti.float32)
A_ti.build_coo(d_row_coo, d_col_coo, d_val_coo)

solver = ti.linalg.SparseSolver()
solver.analyze_pattern(A_ti)
solver.factorize(A_ti)

b = ti.ndarray(shape=nrows, dtype=ti.f32)
b.fill(1.0)

x = ti.ndarray(shape=ncols, dtype=ti.f32)
solver.solve_rf(A_ti, b, x)


@ti.kernel
def print_ndarray(x: ti.types.ndarray()):
    for i in range(10):
        print(x[i])


print_ndarray(x)
