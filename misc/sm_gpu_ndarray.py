import taichi as ti
import numpy as np
ti.init(arch=ti.cuda, gdb_trigger=True)

idx_dt = ti.int32
val_dt = ti.f32
row_csr = ti.ndarray(shape=5, dtype=idx_dt)  
col_csr = ti.ndarray(shape=9, dtype=idx_dt)  
value_csr = ti.ndarray(shape=9, dtype=val_dt)
X = ti.ndarray(shape=4, dtype=val_dt)  
Y = ti.ndarray(shape=4, dtype=val_dt)  
Y_result = ti.ndarray(shape=4, dtype=val_dt)


h_row_csr = np.asarray([ 0, 3, 4, 7, 9], dtype=np.int32)
h_col_csr = np.asarray([0, 2, 3, 1, 0, 2, 3, 1, 3 ], dtype=np.int32)
h_value_csr = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0,6.0, 7.0, 8.0, 9.0], dtype=np.float32)
h_X = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32) 
h_Y_result = np.asarray([19.0, 8.0, 51.0, 52.0], dtype=np.float32)


row_csr.from_numpy(h_row_csr)
col_csr.from_numpy(h_col_csr)
value_csr.from_numpy(h_value_csr)
X.from_numpy(h_X)
Y.fill(0.0)

A = ti.linalg.SparseMatrix(n=4, m=4, dtype=ti.f32)

A.build_from_ndarray_cusparse(row_csr, col_csr, value_csr, X, Y)

for i in range(4):
    print(f"{Y[i]} == {h_Y_result[i]} :  {Y[i] == h_Y_result[i]}")