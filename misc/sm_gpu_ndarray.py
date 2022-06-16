import taichi as ti
import numpy as np
ti.init(arch=ti.cuda)

# Numpy arrays for taichi ndarrays
h_row_csr = np.asarray([ 0, 3, 4, 7, 9], dtype=np.int32)
h_col_csr = np.asarray([0, 2, 3, 1, 0, 2, 3, 1, 3 ], dtype=np.int32)
h_value_csr = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0,6.0, 7.0, 8.0, 9.0], dtype=np.float32)
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
A.build_csr_cusparse(row_csr, col_csr, value_csr)

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