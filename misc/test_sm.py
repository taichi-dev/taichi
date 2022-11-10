import numpy as np

import taichi as ti

ti.init(arch=ti.cuda, debug=True, offline_cache=False)

h_coo_row = np.asarray([1, 0, 0, 0, 2, 2, 2, 3, 3], dtype=np.int32)
h_coo_col = np.asarray([1, 0, 2, 3, 0, 2, 3, 1, 3], dtype=np.int32)
h_coo_val = np.asarray([4.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                       dtype=np.float32)
h_X = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
h_Y = np.asarray([19.0, 8.0, 51.0, 52.0], dtype=np.float32)
# Data structure for building the CSR matrix A using Taichi Sparse Matrix
idx_dt = ti.int32
val_dt = ti.f32
d_coo_row = ti.ndarray(shape=9, dtype=idx_dt)
d_coo_col = ti.ndarray(shape=9, dtype=idx_dt)
d_coo_val = ti.ndarray(shape=9, dtype=val_dt)
# Dense vector x
X = ti.ndarray(shape=4, dtype=val_dt)
# Results for A @ x
# Y = ti.ndarray(shape=4, dtype=val_dt)
# Initialize the CSR matrix and vectors with numpy array
d_coo_row.from_numpy(h_coo_row)
d_coo_col.from_numpy(h_coo_col)
d_coo_val.from_numpy(h_coo_val)
X.from_numpy(h_X)
# Y.fill(0.0)
# Define the CSR matrix A
A = ti.linalg.SparseMatrix(n=4, m=4, dtype=ti.f32)
# Build the CSR matrix A with Taichi ndarray
A.build_coo(d_coo_row, d_coo_col, d_coo_val)
# Compute Y = A @ X
Y = A.spmv(X)
for i in range(4):
    assert Y[i] == h_Y[i]
