import pytest

import taichi as ti
from tests import test_utils


def _test_matrix_slice_read():
    b = 6

    @ti.kernel
    def foo1() -> ti.types.vector(3, dtype=ti.i32):
        c = ti.Vector([0, 1, 2, 3, 4, 5, 6])
        return c[:b:2]

    @ti.kernel
    def foo2() -> ti.types.matrix(2, 3, dtype=ti.i32):
        a = ti.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        return a[1::, :]

    v1 = foo1()
    assert (v1 == ti.Vector([0, 2, 4])).all()
    m1 = foo2()
    assert (m1 == ti.Matrix([[4, 5, 6], [7, 8, 9]])).all()
    v2 = ti.Vector([1, 2, 3, 4, 5, 6])[2::3]
    assert (v2 == ti.Vector([3, 6])).all()
    m2 = ti.Matrix([[2, 3], [4, 5]])[:1, 1:]
    assert (m2 == ti.Matrix([[3]])).all()


@test_utils.test()
def test_matrix_slice_read():
    _test_matrix_slice_read()


@test_utils.test(real_matrix=True, real_matrix_scalarize=True)
def test_matrix_slice_read_real_matrix_scalarize():
    _test_matrix_slice_read()


def _test_matrix_slice_invalid():
    @ti.kernel
    def foo1(i: ti.i32):
        a = ti.Vector([0, 1, 2, 3, 4, 5, 6])
        b = a[i::2]

    @ti.kernel
    def foo2():
        i = 2
        a = ti.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = a[:i:, :i]

    with pytest.raises(ti.TaichiCompilationError,
                       match='Taichi does not support variables in slice now'):
        foo1(1)
    with pytest.raises(ti.TaichiCompilationError,
                       match='Taichi does not support variables in slice now'):
        foo2()


@test_utils.test()
def test_matrix_slice_invalid():
    _test_matrix_slice_invalid()


@test_utils.test(real_matrix=True, real_matrix_scalarize=True)
def test_matrix_slice_invalid_real_matrix_scalarize():
    _test_matrix_slice_invalid()


def _test_matrix_slice_with_variable():
    @ti.kernel
    def test_one_row_slice() -> ti.types.matrix(2, 1, dtype=ti.i32):
        m = ti.Matrix([[1, 2, 3], [4, 5, 6]])
        index = 1
        return m[:, index]

    @ti.kernel
    def test_one_col_slice() -> ti.types.matrix(1, 3, dtype=ti.i32):
        m = ti.Matrix([[1, 2, 3], [4, 5, 6]])
        index = 1
        return m[index, :]

    r1 = test_one_row_slice()
    assert (r1 == ti.Matrix([[2], [5]])).all()
    c1 = test_one_col_slice()
    assert (c1 == ti.Matrix([[4, 5, 6]])).all()


@test_utils.test(dynamic_index=True)
def test_matrix_slice_with_variable():
    _test_matrix_slice_with_variable()


@test_utils.test(real_matrix=True,
                 real_matrix_scalarize=True,
                 dynamic_index=True)
def test_matrix_slice_with_variable_real_matrix_scalarize():
    _test_matrix_slice_with_variable()


@test_utils.test(dynamic_index=False)
def test_matrix_slice_with_variable_invalid():
    @ti.kernel
    def test_one_col_slice() -> ti.types.matrix(1, 3, dtype=ti.i32):
        m = ti.Matrix([[1, 2, 3], [4, 5, 6]])
        index = 1
        return m[index, :]

    with pytest.raises(
            ti.TaichiCompilationError,
            match=
            r'The 0-th index of a Matrix/Vector must be a compile-time constant '
            r"integer, got <class 'taichi.lang.expr.Expr'>.\n"
            r'This is because matrix operations will be \*\*unrolled\*\* at compile-time '
            r'for performance reason.\n'
            r'If you want to \*iterate through matrix elements\*, use a static range:\n'
            r'  for i in ti.static\(range\(3\)\):\n'
            r'    print\(i, "-th component is", vec\[i\]\)\n'
            r'See https://docs.taichi-lang.org/docs/meta#when-to-use-tistatic-with-for-loops for more details.'
            r'Or turn on ti.init\(..., dynamic_index=True\) to support indexing with variables!'
    ):
        test_one_col_slice()


def _test_matrix_slice_write():
    @ti.kernel
    def assign_row() -> ti.types.matrix(3, 4, ti.i32):
        mat = ti.Matrix([[0, 0, 0, 0] for _ in range(3)])
        row = ti.Matrix([[1, 2, 3, 4]])
        mat[0, :] = row
        return mat

    @ti.kernel
    def assign_partial_row() -> ti.types.matrix(3, 4, ti.i32):
        mat = ti.Matrix([[0, 0, 0, 0] for _ in range(3)])
        mat[1, 1:3] = ti.Matrix([[1, 2]])
        return mat

    @ti.kernel
    def augassign_rows() -> ti.types.matrix(3, 4, ti.i32):
        mat = ti.Matrix([[1, 1, 1, 1] for _ in range(3)])
        rows = ti.Matrix([[1, 2, 3, 4] for _ in range(2)])
        mat[:2, :] += rows
        return mat

    assert (assign_row() == ti.Matrix([[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]])).all()
    assert (assign_partial_row() == ti.Matrix([[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]])).all()
    assert (augassign_rows() == ti.Matrix([[2, 3, 4, 5], [2, 3, 4, 5], [1, 1, 1, 1]])).all()


@test_utils.test()
def test_matrix_slice_write():
    _test_matrix_slice_write()


@test_utils.test(real_matrix=True,
                 real_matrix_scalarize=True)
def test_matrix_slice_write_real_matrix_scalarize():
    _test_matrix_slice_write()


def _test_matrix_slice_write_dynamic_index():
    @ti.kernel
    def foo(i: ti.i32) -> ti.types.matrix(3, 4, ti.i32):
        mat = ti.Matrix([[0, 0, 0, 0] for _ in range(3)])
        mat[i, :] = ti.Matrix([[1, 2, 3, 4]])
        return mat

    for i in range(3):
        assert (foo(i) == ti.Matrix([[1, 2, 3, 4] if j == i else [0, 0, 0, 0] for j in range(3)])).all()


@test_utils.test(dynamic_index=True)
def test_matrix_slice_write_dynamic_index():
    _test_matrix_slice_write_dynamic_index()


@test_utils.test(real_matrix=True,
                 real_matrix_scalarize=True,
                 dynamic_index=True)
def test_matrix_slice_write_dynamic_index_real_matrix_scalarize():
    _test_matrix_slice_write_dynamic_index()
