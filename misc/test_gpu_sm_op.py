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

S1 = random(rows, cols, density=0.5, random_state=rng,
            data_rvs=rvs).astype(np_dtype).tocoo()
S2 = random(rows, cols, density=0.5, random_state=rng,
            data_rvs=rvs).astype(np_dtype).tocoo()

nnz_A = len(S1.data)
nnz_B = len(S2.data)

row_coo_A = ti.ndarray(shape=nnz_A, dtype=idx_dt)
col_coo_A = ti.ndarray(shape=nnz_A, dtype=idx_dt)
value_coo_A = ti.ndarray(shape=nnz_A, dtype=val_dt)
row_coo_A.from_numpy(S1.row)
col_coo_A.from_numpy(S1.col)
value_coo_A.from_numpy(S1.data)

row_coo_B = ti.ndarray(shape=nnz_B, dtype=idx_dt)
col_coo_B = ti.ndarray(shape=nnz_B, dtype=idx_dt)
value_coo_B = ti.ndarray(shape=nnz_B, dtype=val_dt)
row_coo_B.from_numpy(S2.row)
col_coo_B.from_numpy(S2.col)
value_coo_B.from_numpy(S2.data)

A = ti.linalg.SparseMatrix(n=rows, m=cols, dtype=ti.f32)
B = ti.linalg.SparseMatrix(n=rows, m=cols, dtype=ti.f32)
A.build_coo(row_coo_A, col_coo_A, value_coo_A)
B.build_coo(row_coo_B, col_coo_B, value_coo_B)

print('>>>> A:')
print(A)
print('>>>> B:')
print(B)

print('>>>> C = A + B:')
C = A + B
print(C)
print('>>>> verify:')
S3 = S1 + S2
print(S3.A)
print('>>>> C - A:')
D = C - A
print(D)
print('>>>> verify:')
print((S3 - S1).A)
print('>>>> A * 2.5:')
E = A * 2.5
print(E)
print('>>>> verify:')
print((S1 * 2.5).A)
print('>>>> A.T:')
F = A.transpose()
print(F)
print('>>>> verify:')
print(S1.T.A)
print('>>>> A @ B.T:')
G = A @ B.transpose()
print(G)
print('>>>> verify:')
print((S1 @ S2.T).A)
