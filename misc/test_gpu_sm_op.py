import numpy as np
import scipy
from numpy.random import default_rng
from scipy import stats
from scipy.sparse import random

import taichi as ti

ti.init(arch=ti.cuda)

idx_dt = ti.i32
val_dt = ti.f32

seed = 2
np.random.seed(seed)

rng = default_rng(seed)
rvs = stats.poisson(3, loc=1).rvs
N = 5
np_dtype = np.float32
rows = N
cols = N - 1

S1 = random(rows, cols, density=0.5, random_state=rng, data_rvs=rvs).astype(np_dtype)
S2 = random(rows, cols, density=0.5, random_state=rng, data_rvs=rvs).astype(np_dtype)
# S2 = S2.T
nnz_A = len(S1.data)
nnz_B = len(S2.data)

coo_S1 = scipy.sparse.coo_matrix(S1)
coo_S2 = scipy.sparse.coo_matrix(S2)

row_coo_A = ti.ndarray(shape=nnz_A, dtype=idx_dt)
col_coo_A = ti.ndarray(shape=nnz_A, dtype=idx_dt)
value_coo_A = ti.ndarray(shape=nnz_A, dtype=val_dt)
row_coo_A.from_numpy(coo_S1.row)
col_coo_A.from_numpy(coo_S1.col)
value_coo_A.from_numpy(coo_S1.data)

row_coo_B = ti.ndarray(shape=nnz_B, dtype=idx_dt)
col_coo_B = ti.ndarray(shape=nnz_B, dtype=idx_dt)
value_coo_B = ti.ndarray(shape=nnz_B, dtype=val_dt)
row_coo_B.from_numpy(coo_S2.row)
col_coo_B.from_numpy(coo_S2.col)
value_coo_B.from_numpy(coo_S2.data)

A = ti.linalg.SparseMatrix(n=rows, m=cols, dtype=ti.f32)
B = ti.linalg.SparseMatrix(n=rows, m=cols, dtype=ti.f32)
A.build_coo(row_coo_A, col_coo_A, value_coo_A)
B.build_coo(row_coo_B, col_coo_B, value_coo_B)

# C = ti.linalg.SparseMatrix(n=rows, m=cols, dtype=ti.f32)
print('>>>> A:')
print(A)
print('>>>> B:')
print(B)

# A.test_destory()

print('>>>> C = A + B:')
C = A + B
print(C)
# print(A)
# print('>>>> C - A:')
# D = C - A
# print(D)
# print('>>>> A * 2.5:')
# E = A * 2.5
# print(E)
print('>>>> A.T:')
G = A.transpose()
print(G)
# print('>>> A @ B.T:')
# F = A @ B
# print(F)
