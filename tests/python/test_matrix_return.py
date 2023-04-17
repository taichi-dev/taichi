import taichi as ti
from tests import test_utils


def _test_vector_return():
    @ti.kernel
    def func() -> ti.types.vector(3, ti.i32):
        return ti.Vector([1, 2, 3])

    assert (func() == ti.Vector([1, 2, 3])).all()


@test_utils.test()
def test_vector_return():
    _test_vector_return()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_vector_return_real_matrix():
    _test_vector_return()


def _test_matrix_return():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i16):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert (func() == ti.Matrix([[1, 2, 3], [4, 5, 6]])).all()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.metal])
def test_matrix_return():
    _test_matrix_return()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_matrix_return_real_matrix():
    _test_matrix_return()


def _test_matrix_return_limit():
    @ti.kernel
    def func() -> ti.types.matrix(3, 10, ti.i32):
        return ti.Matrix(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            ]
        )

    assert (
        func()
        == ti.Matrix(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            ]
        )
    ).all()


@test_utils.test()
def test_matrix_return_limit():
    _test_matrix_return_limit()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_matrix_return_limit_real_matrix():
    _test_matrix_return_limit()
