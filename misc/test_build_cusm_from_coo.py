import numpy as np

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
A.build_coo(d_coo_row, d_coo_col, d_coo_val)

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
