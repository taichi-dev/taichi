import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_slice():
    b = 3

    @ti.kernel
    def foo1() -> ti.types.vector(3, dtype=ti.i32):
        c = ti.Vector([0, 1, 2, 3, 4, 5, 6])
        return c[:5:2]

    @ti.kernel
    def foo2() -> ti.types.matrix(2, 2, dtype=ti.i32):
        a = ti.Matrix([[1, 2, 3], [4, 5, 6]])
        return a[:, :b:2]

    v1 = foo1()
    assert (v1 == ti.Vector([0, 2, 4])).all() == 1
    m1 = foo2()
    assert (m1 == ti.Matrix([[1, 3], [4, 6]])).all() == 1


@test_utils.test(dynamic_index=True)
def test_dyn():
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
    assert (r1 == ti.Matrix([[2], [5]])).all() == 1
    c1 = test_one_col_slice()
    assert (c1 == ti.Matrix([[4, 5, 6]])).all() == 1


@test_utils.test(dynamic_index=False)
def test_no_dyn():
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
