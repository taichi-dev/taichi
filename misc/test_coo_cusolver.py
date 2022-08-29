import urllib.request

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


print(">> downloading sparse matrix lap2D_5pt_n100.mtx...")
url = 'https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Samples/4_CUDA_Libraries/cuSolverSp_LinearSolver/lap2D_5pt_n100.mtx'
urllib.request.urlretrieve(url, 'misc/lap2D_5pt_n100.mtx')
print(">> load sparse matrix........")
A_raw_coo = sio.mmread('misc/lap2D_5pt_n100.mtx')
nrows, ncols = A_raw_coo.shape
nnz = A_raw_coo.nnz

A_csr = A_raw_coo.tocsr()
b = ti.ndarray(shape=nrows, dtype=ti.f32)
init_b(b, nrows)

print(">> solve Ax = b using CuSparseSolver ......... ")
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
print(">> CuSparseSolver results:")
print_x(x_ti, ncols, 10)
