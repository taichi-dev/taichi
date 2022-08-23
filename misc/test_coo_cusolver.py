import numpy as np
import scipy.io as sio

import taichi as ti

ti.init(arch=ti.cuda)


@ti.kernel
def init_b(b: ti.types.ndarray(), nrows: ti.i32):
    for i in range(nrows):
        b[i] = 1.0 + i / nrows


@ti.kernel
def print_x(x: ti.types.ndarray(), ncols: ti.i32, length: ti.i32):
    for i in range(ncols - length, ncols):
        print(x[i])


print(">> load sparse matrix........")
A_raw_coo = sio.mmread('misc/lap2D_5pt_n100.mtx')
nrows, ncols = A_raw_coo.shape
nnz = A_raw_coo.nnz

A_csr = A_raw_coo.tocsr()

d_row_csr = ti.ndarray(shape=nrows + 1, dtype=ti.i32)
d_col_csr = ti.ndarray(shape=nnz, dtype=ti.i32)
d_value_csr = ti.ndarray(shape=nnz, dtype=ti.f32)
d_row_csr.from_numpy(A_csr.indptr)
d_col_csr.from_numpy(A_csr.indices)
d_value_csr.from_numpy(A_csr.data)
# solve Ax = b using csr format ndarrays
print(">> solve Ax = b using cu_solve() ......... ")
b = ti.ndarray(shape=nrows, dtype=ti.f32)
x = ti.ndarray(shape=ncols, dtype=ti.f32)
init_b(b, nrows)
ti.linalg.cu_solve(d_row_csr, d_col_csr, d_value_csr, nrows, ncols, nnz, b, x)
ti.sync()
print(">> cusolve result:")
print_x(x, ncols, 10)

ti.sync()
print(">> solve Ax = b using CuSparseSolver ......... ")
A_coo = A_csr.tocoo()
d_row_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
d_col_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
d_val_coo = ti.ndarray(shape=nnz, dtype=ti.f32)
d_row_coo.from_numpy(A_coo.row)
d_col_coo.from_numpy(A_coo.col)
d_val_coo.from_numpy(A_coo.data)

A_ti = ti.linalg.SparseMatrix(n=nrows, m=ncols, dtype=ti.float32)
A_ti.build_csr_cusparse(d_row_coo, d_col_coo, d_val_coo)
x_ti = ti.ndarray(shape=ncols, dtype=ti.float32)
solver = ti.linalg.SparseSolver()
solver.solve_cu(A_ti, b, x_ti)
ti.sync()
print(">> CuSparseSolver results:")
print_x(x_ti, ncols, 10)
