---
sidebar_position: 3
---

# Linear Solver

Solving linear equations is a common task in scientific computing. Taichi provides basic direct and iterative linear solvers for
various simulation scenarios. Currently, there are two categories of linear solvers available:
1. Solvers built for `SparseMatrix`, including:
- Direct solver `SparseSolver`
- Iterative (conjugate-gradient method) solver `SparseCG`
2. Solvers built for `LinearOperator`
- Iterative (matrix-free conjugate-gradient method) solver `MatrixfreeCG`

It's important to understand that those solvers are built for specific types of matrices. For example, if you built a coefficient matrix `A` as a `SparseMatrix`, then you can only use `SparseSolver` or `SparseCG` to solve the corresponding linear system. Below we will explain the usage of each type of solvers.

## Sparse linear solver
There are two types of linear solvers available for `SparseMatrix`, direct solver and iterative solver.

### Sparse direct solver
To solve a linear system whose coefficient matrix is a `SparseMatrix` using a direct method, follow the steps below:
1. Create a `solver` using `ti.linalg.SparseSolver(solver_type, ordering)`. Currently, the factorization types supported on CPU backends are `LLT`, `LDLT`, and `LU`, and supported orderings include `AMD` and `COLAMD`. The sparse solver on CUDA supports the `LLT` factorization type only.
2. Analyze and factorize the sparse matrix you want to solve using `solver.analyze_pattern(sparse_matrix)` and `solver.factorize(sparse_matrix)`
3. Call `x = solver.solve(b)`, where `x` is the solution and `b` is the right-hand side of the linear system. On CPU backends, `x` and `b` can be NumPy arrays, Taichi Ndarrays, or Taichi fields. On the CUDA backend, `x` and `b` *must* be Taichi Ndarrays.
4. Call `solver.info()` to check if the solving process succeeds.

Here's a full example.

```python
import taichi as ti

arch = ti.cpu # or ti.cuda
ti.init(arch=arch)

n = 4

K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
b = ti.ndarray(ti.f32, shape=n)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray(), interval: ti.i32):
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
success = solver.info()
print(">>>> Solve sparse linear systems Ax = b with the solution x:")
print(x)
print(f">>>> Computation succeed: {success}")
# outputs:
# >>>> Solve sparse linear systems Ax = b with the solution x:
# [0.5 0.  0.  0.5]
# >>>> Computation was successful?: True
```

Please have a look at our two demos for more information:
+ [Stable fluid](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py): A 2D fluid simulation using a sparse Laplacian matrix to solve Poisson's pressure equation.
+ [Implicit mass spring](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/implicit_mass_spring.py): A 2D cloth simulation demo using sparse matrices to solve the linear systems.

### Sparse iterative solver
To solve a linear system whose coefficient matrix is a `SparseMatrix` using a iterative (conjugate-gradient) method, follow the steps below:
1. Create a `solver` using `ti.linalg.SparseCG(A, b, x0, max_iter, atol)`, where `A` is a `SparseMatrix` that stores the coefficient matrix of the linear system, `b` is the right-hand side of the equations, `x0` is the initial guess and `atol` is the absolute tolerance threshold.
2. Call `x, exit_code = solver.solve()` to obtain the solution `x` along with the `exit_code` that indicates the status of the solution. `exit_code` should be `True` if the solving was successful. Here is an example:

```python
import taichi as ti

ti.init(arch=ti.cpu)

n = 4

K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
b = ti.ndarray(ti.f32, shape=n)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray(), interval: ti.i32):
    for i in range(n):
        A[i, i] += 2.0
        if i % interval == 0:
            b[i] += 1.0

fill(K, b, 3)

A = K.build()
print(">>>> Matrix A:")
print(A)
print(">>>> Vector b:")
print(b.to_numpy())
# outputs:
# >>>> Matrix A:
# [2, 0, 0, 0]
# [0, 2, 0, 0]
# [0, 0, 2, 0]
# [0, 0, 0, 2]
# >>>> Vector b:
# [1. 0. 0. 1.]
solver = ti.linalg.SparseCG(A, b)
x, exit_code = solver.solve()
print(">>>> Solve sparse linear systems Ax = b with the solution x:")
print(x)
print(f">>>> Computation was successful?: {exit_code}")
# outputs:
# >>>> Solve sparse linear systems Ax = b with the solution x:
# [0.5 0.  0.  0.5]
# >>>> Computation was successful?: True
```
Note that the building process of `SparseMatrix` `A` is exactly the same as in the case of `SparseSolver`, the only difference here is that we created a `solver` whose type is `SparseCG` instead of `SparseSolver`.

## Matrix-free iterative solver
Apart from `SparseMatrix` as an efficient representation of matrices, Taichi also support the `LinearOperator` type, which is a matrix-free representation of matrices.
Keep in mind that matrices can be seen as a linear transformation from an input vector to a output vector, it is possible to encapsulate the information of a matrice as a `LinearOperator`.

To create a `LinearOperator` in Taichi, we first need to define a kernel that represent the linear transformation:
```python
import taichi as ti
from taichi.linalg import LinearOperator

ti.init(arch=ti.cpu)

@ti.kernel
def compute_matrix_vector(v:ti.template(), mv:ti.template()):
    for i in v:
        mv[i] = 2 * v[i]
```
In this case, `compute_matrix_vector` kernel accepts an input vector `v` and calculates the corresponding matrix-vector product `mv`. It is mathematically equal to a matrice whose diagonal elements are all 2. In the case of `n=4`, the equivalent matrice `A` is:
```python cont
# >>>> Matrix A:
# [2, 0, 0, 0]
# [0, 2, 0, 0]
# [0, 0, 2, 0]
# [0, 0, 0, 2]
```
Then we can create the `LinearOperator` as follows:
```python cont
A = LinearOperator(compute_matrix_vector)
```
To solve a system of linear equations represented by this `LinearOperator`, we can use the built-in matrix-free solver `MatrixFreeCG`. Here is a full example:

```python
import taichi as ti
import math
from taichi.linalg import MatrixFreeCG, LinearOperator

ti.init(arch=ti.cpu)

GRID = 4
Ax = ti.field(dtype=ti.f32, shape=(GRID, GRID))
x = ti.field(dtype=ti.f32, shape=(GRID, GRID))
b = ti.field(dtype=ti.f32, shape=(GRID, GRID))

@ti.kernel
def init():
    for i, j in ti.ndrange(GRID, GRID):
        xl = i / (GRID - 1)
        yl = j / (GRID - 1)
        b[i, j] = ti.sin(2 * math.pi * xl) * ti.sin(2 * math.pi * yl)
        x[i, j] = 0.0

@ti.kernel
def compute_Ax(v: ti.template(), mv: ti.template()):
    for i, j in v:
        l = v[i - 1, j] if i - 1 >= 0 else 0.0
        r = v[i + 1, j] if i + 1 <= GRID - 1 else 0.0
        t = v[i, j + 1] if j + 1 <= GRID - 1 else 0.0
        b = v[i, j - 1] if j - 1 >= 0 else 0.0
        # Avoid ill-conditioned matrix A
        mv[i, j] = 20 * v[i, j] - l - r - t - b

A = LinearOperator(compute_Ax)
init()
MatrixFreeCG(A, b, x, maxiter=10 * GRID * GRID, tol=1e-18, quiet=True)
print(x.to_numpy())
```
