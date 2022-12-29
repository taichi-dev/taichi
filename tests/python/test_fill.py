from taichi.lang import impl

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_fill_scalar_field():
    n = 4
    m = 7
    val = ti.field(ti.i32, shape=(n, m))

    val.fill(2)
    for i in range(n):
        for j in range(m):
            assert val[i, j] == 2

    @ti.kernel
    def fill_in_kernel(v: ti.i32):
        val.fill(v)

    fill_in_kernel(3)
    for i in range(n):
        for j in range(m):
            assert val[i, j] == 3


@test_utils.test()
def test_fill_matrix_field_with_scalar():
    n = 4
    m = 7
    val = ti.Matrix.field(2, 3, ti.i32, shape=(n, m))

    val.fill(2)
    for i in range(n):
        for j in range(m):
            assert (val[i, j] == 2).all()

    @ti.kernel
    def fill_in_kernel(v: ti.i32):
        val.fill(v)

    fill_in_kernel(3)
    for i in range(n):
        for j in range(m):
            assert (val[i, j] == 3).all()


@test_utils.test()
def test_fill_matrix_field_with_matrix():
    n = 4
    m = 7
    val = ti.Matrix.field(2, 3, ti.i32, shape=(n, m))

    mat = ti.Matrix([[0, 1, 2], [2, 3, 4]])
    val.fill(mat)
    for i in range(n):
        for j in range(m):
            assert (val[i, j] == mat).all()

    @ti.kernel
    def fill_in_kernel(v: ti.types.matrix(2, 3, ti.i32)):
        val.fill(v)

    mat = ti.Matrix([[4, 5, 6], [6, 7, 8]])
    fill_in_kernel(mat)
    for i in range(n):
        for j in range(m):
            assert (val[i, j] == mat).all()


@test_utils.test()
def test_fill_vector_field_recompile():
    a = ti.Vector.field(2, ti.i32, shape=3)
    for i in range(2):
        a.fill(ti.Vector([0, 0]))
    assert impl.get_runtime().get_num_compiled_functions() == 1
