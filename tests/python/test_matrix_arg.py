import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_matrix_arg():
    mat1 = ti.Matrix([[1, 2, 3], [4, 5, 6]])

    @ti.kernel
    def foo(mat: ti.types.matrix(2, 3, ti.i32)) -> ti.i32:
        return mat[0, 0] + mat[1, 2]

    assert foo(mat1) == 7

    mat3 = ti.Matrix([[1, 2], [3, 4], [5, 6]])

    @ti.kernel
    def foo2(var: ti.i32, mat: ti.types.matrix(3, 2, ti.i32)) -> ti.i32:
        for i in ti.static(range(3)):
            for j in ti.static(range(2)):
                mat[i, j] += var
        return mat[2, 1]

    assert foo2(3, mat3) == 9


@test_utils.test()
def test_vector_arg():
    vec1 = ti.Vector([1, 2, 3])

    @ti.kernel
    def foo(vec: ti.types.vector(3, ti.i32)) -> int:
        return vec[0] + vec[1] + vec[2]

    assert foo(vec1) == 6
