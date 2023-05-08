---
sidebar_position: 3
---

# Linear Solver

Solving linear equations is a common task in scientific computing. Taichi provides basic direct and iterative linear solvers for
various simulation scenarios. Currently, there are two categories of linear solvers available:
1. Solvers built for `SparseMatrix`
2. Solvers built for `ti.field`

## Sparse linear solver
You may want to solve some linear equations using sparse matrices.
Then, the following steps could help:
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
## Examples

Please have a look at our two demos for more information:
+ [Stable fluid](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py): A 2D fluid simulation using a sparse Laplacian matrix to solve Poisson's pressure equation.
+ [Implicit mass spring](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/implicit_mass_spring.py): A 2D cloth simulation demo using sparse matrices to solve the linear systems.
