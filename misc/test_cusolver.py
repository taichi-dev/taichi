import numpy as np
import scipy.io as sio

import taichi as ti

ti.init(arch=ti.cuda)

h_coo_row = np.asarray([0, 0, 0, 1, 2, 2, 2, 3, 3], dtype=np.int32)
h_coo_col = np.asarray([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
h_coo_val = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                       dtype=np.float32)
h_x = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
h_y = np.asarray([19.0, 8.0, 51.0, 52.0], dtype=np.float32)

d_coo_row = ti.ndarray(shape=9, dtype=ti.int32)
d_coo_col = ti.ndarray(shape=9, dtype=ti.int32)
d_coo_val = ti.ndarray(shape=9, dtype=ti.float32)
x = ti.ndarray(shape=4, dtype=ti.float32)
y = ti.ndarray(shape=4, dtype=ti.float32)

d_coo_row.from_numpy(h_coo_row)
d_coo_col.from_numpy(h_coo_col)
d_coo_val.from_numpy(h_coo_val)
x.from_numpy(h_x)
y.fill(0.0)

A = ti.linalg.SparseMatrix(n=4, m=4, dtype=ti.float32)
A.build_csr_cusparse(d_coo_val, d_coo_col, d_coo_row)

A.spmv(x, y)

# Check if the results are correct
equal = True
for i in range(4):
    if y[i] != h_y[i]:
        equal = False
        break
if equal:
    print("Spmv Results is correct!")
else:
    print("Opps! Spmv Results is wrong.")

print("""----------------- sparse solver -----------------""")


@ti.kernel
def init_b(b: ti.types.ndarray()):
    for i in range(nrows):
        b[i] = 1.0 + i / nrows


@ti.kernel
def print_x(x: ti.types.ndarray()):
    for i in range(ncols - 10, ncols):
        print(x[i])


# Read .mtx file
A_coo = sio.mmread('misc/lap2D_5pt_n100.mtx')
nrows, ncols = A_coo.shape
nnz = A_coo.nnz
A_ti = ti.linalg.SparseMatrix(n=nrows, m=ncols, dtype=ti.float32)
d_val_coo = ti.ndarray(shape=nnz, dtype=ti.float32)
d_col_coo = ti.ndarray(shape=nnz, dtype=ti.int32)
d_row_coo = ti.ndarray(shape=nnz, dtype=ti.int32)
d_val_coo.from_numpy(A_coo.data)
d_col_coo.from_numpy(A_coo.col)
d_row_coo.from_numpy(A_coo.row)

# build csr format for ti.linalg.SparseMatrix
A_ti.build_csr_cusparse(d_val_coo, d_col_coo, d_row_coo)

# Initialize b
b = ti.ndarray(shape=nrows, dtype=ti.f32)
init_b(b)

# Initialize x
x = ti.ndarray(shape=ncols, dtype=ti.f32)
x.fill(0.0)

A_csr = A_coo.tocsr()
d_col_csr = ti.ndarray(shape=nnz, dtype=ti.i32)
d_value_csr = ti.ndarray(shape=nnz, dtype=ti.f32)
d_row_csr = ti.ndarray(shape=nrows + 1, dtype=ti.i32)
d_value_csr.from_numpy(A_csr.data)
d_col_csr.from_numpy(A_csr.indices)
d_row_csr.from_numpy(A_csr.indptr)
# solve Ax = b using csr format ndarrays
ti.linalg.cu_solve(d_row_csr, d_col_csr, d_value_csr, nrows, ncols, nnz, b, x)
print_x(x)

# solve A_ti * x = b
solver = ti.linalg.SparseSolver(solver_type="LLT")
# solver.solve(A_ti, x, b)
