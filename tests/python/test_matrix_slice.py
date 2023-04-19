import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def _test_matrix_slice_read_python_scope():
    v1 = ti.Vector([1, 2, 3, 4, 5, 6])[2::3]
    assert (v1 == ti.Vector([3, 6])).all()
    m = ti.Matrix([[2, 3], [4, 5]])[:1, 1:]
    assert (m == ti.Matrix([[3]])).all()
    v2 = ti.Matrix([[1, 2], [3, 4]])[:, 1]
    assert (v2 == ti.Vector([2, 4])).all()


@test_utils.test()
def test_matrix_slice_read():
    b = 6

    @ti.kernel
    def foo1() -> ti.types.vector(3, dtype=ti.i32):
        c = ti.Vector([0, 1, 2, 3, 4, 5, 6])
        return c[:b:2]

    @ti.kernel
    def foo2() -> ti.types.matrix(2, 3, dtype=ti.i32):
        a = ti.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        return a[1::, :]

    v = foo1()
    assert (v == ti.Vector([0, 2, 4])).all()
    m = foo2()
    assert (m == ti.Matrix([[4, 5, 6], [7, 8, 9]])).all()


@test_utils.test()
def test_matrix_slice_invalid():
    @ti.kernel
    def foo1(i: ti.i32):
        a = ti.Vector([0, 1, 2, 3, 4, 5, 6])
        b = a[i::2]

    @ti.kernel
    def foo2():
        i = 2
        a = ti.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = a[:i:, :i]

    with pytest.raises(
        ti.TaichiCompilationError,
        match="Taichi does not support variables in slice now",
    ):
        foo1(1)
    with pytest.raises(
        ti.TaichiCompilationError,
        match="Taichi does not support variables in slice now",
    ):
        foo2()


@test_utils.test()
def test_matrix_slice_with_variable():
    @ti.kernel
    def test_one_row_slice(index: ti.i32) -> ti.types.vector(2, dtype=ti.i32):
        m = ti.Matrix([[1, 2, 3], [4, 5, 6]])
        return m[:, index]

    @ti.kernel
    def test_one_col_slice(index: ti.i32) -> ti.types.vector(3, dtype=ti.i32):
        m = ti.Matrix([[1, 2, 3], [4, 5, 6]])
        return m[index, :]

    r1 = test_one_row_slice(1)
    assert (r1 == ti.Vector([2, 5])).all()
    c1 = test_one_col_slice(1)
    assert (c1 == ti.Vector([4, 5, 6])).all()


@test_utils.test()
def test_matrix_slice_write():
    @ti.kernel
    def assign_col() -> ti.types.matrix(3, 4, ti.i32):
        mat = ti.Matrix([[0, 0, 0, 0] for _ in range(3)])
        col = ti.Vector([1, 2, 3])
        mat[:, 0] = col
        return mat

    @ti.kernel
    def assign_partial_row() -> ti.types.matrix(3, 4, ti.i32):
        mat = ti.Matrix([[0, 0, 0, 0] for _ in range(3)])
        mat[1, 1:3] = ti.Vector([1, 2])
        return mat

    @ti.kernel
    def augassign_rows() -> ti.types.matrix(3, 4, ti.i32):
        mat = ti.Matrix([[1, 1, 1, 1] for _ in range(3)])
        rows = ti.Matrix([[1, 2, 3, 4] for _ in range(2)])
        mat[:2, :] += rows
        return mat

    assert (assign_col() == ti.Matrix([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]])).all()
    assert (assign_partial_row() == ti.Matrix([[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]])).all()
    assert (augassign_rows() == ti.Matrix([[2, 3, 4, 5], [2, 3, 4, 5], [1, 1, 1, 1]])).all()


@test_utils.test()
def test_matrix_slice_write_dynamic_index():
    @ti.kernel
    def foo(i: ti.i32) -> ti.types.matrix(3, 4, ti.i32):
        mat = ti.Matrix([[0, 0, 0, 0] for _ in range(3)])
        mat[i, :] = ti.Vector([1, 2, 3, 4])
        return mat

    for i in range(3):
        assert (foo(i) == ti.Matrix([[1, 2, 3, 4] if j == i else [0, 0, 0, 0] for j in range(3)])).all()
