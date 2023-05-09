import numpy as np
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
        res = mat
        for i in ti.static(range(3)):
            for j in ti.static(range(2)):
                res[i, j] += var
        return res[2, 1]

    assert foo2(3, mat3) == 9


@test_utils.test()
def test_vector_arg():
    vec1 = ti.Vector([1, 2, 3])

    @ti.kernel
    def foo(vec: ti.types.vector(3, ti.i32)) -> int:
        return vec[0] + vec[1] + vec[2]

    assert foo(vec1) == 6


@test_utils.test()
def test_matrix_fancy_arg():
    from taichi.math import mat3, vec3

    mat4x3 = ti.types.matrix(4, 3, float)
    mat2x6 = ti.types.matrix(2, 6, float)

    a = np.random.random(3)
    b = np.random.random((3, 3))

    v = vec3(0, 1, 2)
    v = vec3([0, 1, 2])

    M = mat3(a, a, a)
    M = mat3(b)

    m = mat4x3([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])

    m = mat4x3([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    m = mat4x3(vec3(1, 2, 3), vec3(4, 5, 6), vec3(7, 8, 9), vec3(10, 11, 12))

    m = mat4x3([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    m = mat4x3(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    m = mat4x3(1)

    m = mat4x3(m)

    k = mat2x6(m)


@test_utils.test()
def test_matrix_arg_insertion_pos():
    rgba8 = ti.types.vector(4, ti.u8)

    @ti.kernel
    def _render(
        color_attm: ti.types.ndarray(rgba8, ndim=2),
        camera_pos: ti.math.vec3,
        camera_up: ti.math.vec3,
    ):
        up = ti.math.normalize(camera_up)

        for x, y in color_attm:
            o = camera_pos

    color_attm = ti.Vector.ndarray(4, dtype=ti.u8, shape=(512, 512))
    camera_pos = ti.math.vec3(0, 0, 0)
    camera_up = ti.math.vec3(0, 1, 0)

    _render(color_attm, camera_pos, camera_up)
