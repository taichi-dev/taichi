import taichi as ti
import numpy as np
ti.init(arch=ti.cuda)


row_csr = ti.ndarray(shape=5, dtype=ti.f32)  
col_csr = ti.ndarray(shape=9, dtype=ti.f32)  
value_csr = ti.ndarray(shape=9, dtype=ti.f32)

h_row_csr = np.asarray([ 0, 3, 4, 7, 9])
h_col_csr = np.asarray([0, 2, 3, 1, 0, 2, 3, 1, 3 ])
h_value_csr = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0,6.0, 7.0, 8.0, 9.0])


for i in range(5):
    row_csr[i] = h_row_csr[i]
for i in range(9):
    col_csr[i] = h_col_csr[i]
for i in range(9):
    value_csr[i] = h_value_csr[i]



A = ti.linalg.SparseMatrix(n=4, m=4, dtype=ti.f32)

A.build_from_ndarray_cusparse(row_csr, col_csr, value_csr)