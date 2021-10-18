import numpy as np
import pytest

import taichi as ti

"""
The symmetric positive definite matrix is created in matlab using the following script:
    A = diag([1,2,3,4]);
    OrthM = [1 0 1 0; -1 -2 0 1; 0 1 -1 0; 0, 1, 0 1];
    U = orth(OrthM);
    Aarray = U * A * U';
    b = [1,2,3,4]';
    res = inv(A) * b;
"""
Aarray = np.array([[
    2.73999501130921, 0.518002544441220, 0.745119303009342, 0.0508907745638859
], [0.518002544441220, 1.45111665837647, 0.757997555750432, 0.290885785873098],
                   [
                       0.745119303009342, 0.757997555750432, 2.96711176987733,
                       -0.518002544441220
                   ],
                   [
                       0.0508907745638859, 0.290885785873098,
                       -0.518002544441220, 2.84177656043698
                   ]])

res = np.array([
    -0.0754984396447588, 0.469972700892492, 1.18527357933586, 1.57686870529319
])


@pytest.mark.parametrize("solver_type", ["LLT", "LDLT", "LU"])
@ti.test(arch=ti.cpu)
def test_sparse_LLT_solver(solver_type):
    n = 4
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
    b = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.linalg.sparse_matrix_builder(),
             InputArray: ti.ext_arr(), b: ti.template()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, Aarray, b)
    A = Abuilder.build()
    solver = ti.linalg.SparseSolver(solver_type=solver_type)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)
    for i in range(n):
        assert x[i] == ti.approx(res[i])
