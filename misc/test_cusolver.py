import numpy as np
import scipy.io as sio

import taichi as ti

ti.init(arch=ti.cuda)

# Numpy arrays for taichi ndarrays
h_row_csr = np.asarray([0, 3, 4, 7, 9], dtype=np.int32)
h_col_csr = np.asarray([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
h_value_csr = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                         dtype=np.float32)
h_X = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
h_Y = np.asarray([19.0, 8.0, 51.0, 52.0], dtype=np.float32)

# Data structure for building the CSR matrix A using Taichi Sparse Matrix
idx_dt = ti.int32
val_dt = ti.f32
row_csr = ti.ndarray(shape=5, dtype=idx_dt)
col_csr = ti.ndarray(shape=9, dtype=idx_dt)
value_csr = ti.ndarray(shape=9, dtype=val_dt)
# Dense vector x
X = ti.ndarray(shape=4, dtype=val_dt)
# Results for A @ x
Y = ti.ndarray(shape=4, dtype=val_dt)

# Initialize the CSR matrix and vectors with numpy array
row_csr.from_numpy(h_row_csr)
col_csr.from_numpy(h_col_csr)
value_csr.from_numpy(h_value_csr)
X.from_numpy(h_X)
Y.fill(0.0)

# Define the CSR matrix A
A = ti.linalg.SparseMatrix(n=4, m=4, dtype=ti.f32)

# Build the CSR matrix A with Taichi ndarray
A.build_csr_cusparse(value_csr, col_csr, row_csr)

# Compute Y = A @ X
A.spmv(X, Y)

# Check if the results are correct
equal = True
for i in range(4):
    if Y[i] != h_Y[i]:
        equal = False
        break
if equal:
    print("Spmv Results is correct!")
else:
    print("Opps! Spmv Results is wrong.")

solver = ti.linalg.SparseSolver(solver_type="LLT")

# solver.solve(A, X, Y)

# Read .mtx file
A_coo = sio.mmread('misc/lap2D_5pt_n100.mtx')
A_csr = A_coo.tocsr()

nnz = A_csr.nnz
nrows, ncols = A_csr.shape
d_col_csr = ti.ndarray(shape=nnz, dtype=ti.i32)
d_value_csr = ti.ndarray(shape=nnz, dtype=ti.f32)
d_row_csr = ti.ndarray(shape=nrows + 1, dtype=ti.i32)

d_value_csr.from_numpy(A_csr.data)
d_col_csr.from_numpy(A_csr.indices)
d_row_csr.from_numpy(A_csr.indptr)

b = ti.ndarray(shape=nrows, dtype=ti.f32)
b.fill(1.0)

x = ti.ndarray(shape=ncols, dtype=ti.f32)
x.fill(0.0)


@ti.kernel
def init_b(b: ti.types.ndarray()):
    for i in range(nrows):
        b[i] = 1.0 + i / nrows


ti.linalg.cu_solve(d_row_csr, d_col_csr, d_value_csr, nrows, ncols, nnz, b, x)


@ti.kernel
def print_x(x: ti.types.ndarray()):
    for i in range(ncols - 10, ncols):
        print(x[i])


init_b(b)
print_x(x)
