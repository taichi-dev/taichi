---
sidebar_position: 2
---

# Sparse Matrix

Sparse matrices are frequently involved in solving linear systems in science and engineering. Taichi provides useful APIs for sparse matrices on the CPU and CUDA backends.

To use sparse matrices in Taichi programs, follow these three steps:

1. Create a `builder` using `ti.linalg.SparseMatrixBuilder()`.
2. Call `ti.kernel` to fill the `builder` with your matrices' data.
3. Build sparse matrices from the `builder`.

:::caution WARNING
The sparse matrix feature is still under development. There are some limitations:
- The sparse matrix data type on the CPU backend only supports `f32` and `f64`.
- The sparse matrix data type on the CUDA backend only supports `f32`.

:::
Here's an example:
```python
import taichi as ti
arch = ti.cpu # or ti.cuda
ti.init(arch=arch)

n = 4
# step 1: create sparse matrix builder
K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    for i in range(n):
        A[i, i] += 1  # Only +=  and -= operators are supported for now.

# step 2: fill the builder with data.
fill(K)

print(">>>> K.print_triplets()")
K.print_triplets()
# outputs:
# >>>> K.print_triplets()
# n=4, m=4, num_triplets=4 (max=100)(0, 0) val=1.0(1, 1) val=1.0(2, 2) val=1.0(3, 3) val=1.0

# step 3: create a sparse matrix from the builder.
A = K.build()
print(">>>> A = K.build()")
print(A)
# outputs:
# >>>> A = K.build()
# [1, 0, 0, 0]
# [0, 1, 0, 0]
# [0, 0, 1, 0]
# [0, 0, 0, 1]
```

The basic operations like `+`, `-`, `*`, `@` and transpose of sparse matrices are supported now.

```python cont
print(">>>> Summation: C = A + A")
C = A + A
print(C)
# outputs:
# >>>> Summation: C = A + A
# [2, 0, 0, 0]
# [0, 2, 0, 0]
# [0, 0, 2, 0]
# [0, 0, 0, 2]

print(">>>> Subtraction: D = A - A")
D = A - A
print(D)
# outputs:
# >>>> Subtraction: D = A - A
# [0, 0, 0, 0]
# [0, 0, 0, 0]
# [0, 0, 0, 0]
# [0, 0, 0, 0]

print(">>>> Multiplication with a scalar on the right: E = A * 3.0")
E = A * 3.0
print(E)
# outputs:
# >>>> Multiplication with a scalar on the right: E = A * 3.0
# [3, 0, 0, 0]
# [0, 3, 0, 0]
# [0, 0, 3, 0]
# [0, 0, 0, 3]

print(">>>> Multiplication with a scalar on the left: E = 3.0 * A")
E = 3.0 * A
print(E)
# outputs:
# >>>> Multiplication with a scalar on the left: E = 3.0 * A
# [3, 0, 0, 0]
# [0, 3, 0, 0]
# [0, 0, 3, 0]
# [0, 0, 0, 3]

print(">>>> Transpose: F = A.transpose()")
F = A.transpose()
print(F)
# outputs:
# >>>> Transpose: F = A.transpose()
# [1, 0, 0, 0]
# [0, 1, 0, 0]
# [0, 0, 1, 0]
# [0, 0, 0, 1]

print(">>>> Matrix multiplication: G = E @ A")
G = E @ A
print(G)
# outputs:
# >>>> Matrix multiplication: G = E @ A
# [3, 0, 0, 0]
# [0, 3, 0, 0]
# [0, 0, 3, 0]
# [0, 0, 0, 3]

print(">>>> Element-wise multiplication: H = E * A")
H = E * A
print(H)
# outputs:
# >>>> Element-wise multiplication: H = E * A
# [3, 0, 0, 0]
# [0, 3, 0, 0]
# [0, 0, 3, 0]
# [0, 0, 0, 3]

print(f">>>> Element Access: A[0,0] = {A[0,0]}")
# outputs:
# >>>> Element Access: A[0,0] = 1.0
```
