# Sparse Matrix
Sparse matrices are frequently used when solving linear systems in science and engineering. Taichi provides programmers with useful APIs for sparse matrices.

To use the sparse matrix in taichi programs, you should follow these three steps:
1. Create a `builder` using `ti.linalg.SparseMatrixBuilder()`.
2. Fill the `builder` with your matrices' data.
3. Create sparse matrices from the `builder`.

:::caution WARNING
The sparse matrix is still under implementation. There are some limitations:
- Only the CPU backend is supported.
- The data type of sparse matrix is float32.
- The storage format is column-major
:::
Here's an example:
```python
import taichi as ti
ti.init(arch=ti.x64) # only CPU backend is supported for now

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

```python
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

## Sparse linear solver
You may want to solve some linear equations using sparse matrices.
Then, the following steps could help:
1. Create a `solver` using `ti.linalg.SparseSolver(solver_type, ordering)`. Currently, the sparse solver supports `LLT`, `LDLT` and `LU` factorization types, and orderings including `AMD`, `COLAMD`.
2. Analyze and factorize the sparse matrix you want to solve using `solver.analyze_pattern(sparse_matrix)` and `solver.factorize(sparse_matrix)`
3. Call `solver.solve(b)` to get your solutions, where `b` is a numpy array or taichi filed representing the right-hand side of the linear system.
4. Call `solver.info()` to check if the solving process succeeds.

Here's a full example.

```python
import taichi as ti

ti.init(arch=ti.x64)

n = 4

K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
b = ti.field(ti.f32, shape=n)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), b: ti.template(), interval: ti.i32):
    for i in range(n):
        A[i, i] += 2.0

        if i % interval == 0:
            b[i] += 1.0

fill(K, b, 3)

A = K.build()
print(">>>> Matrix A:")
print(A)
print(">>>> Vector b:")
print(b)
# outputs:
# >>>> Matrix A:
# [2, 0, 0, 0]
# [0, 2, 0, 0]
# [0, 0, 2, 0]
# [0, 0, 0, 2]
# >>>> Vector b:
# [1. 0. 0. 1.]
solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(A)
solver.factorize(A)
x = solver.solve(b)
isSuccess = solver.info()
print(">>>> Solve sparse linear systems Ax = b with the solution x:")
print(x)
print(f">>>> Computation was successful?: {isSuccess}")
# outputs:
# >>>> Solve sparse linear systems Ax = b with the solution x:
# [0.5 0.  0.  0.5]
# >>>> Computation was successful?: True
```
## Examples

Please have a look at our two demos for more information:
+ [Stable fluid](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py): A 2D fluid simulation using a sparse Laplacian matrix to solve Poisson's pressure equation.
+ [Implicit mass spring](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/implicit_mass_spring.py): A 2D cloth simulation demo using sparse matrices to solve the linear systems.
