import numbers

import taichi.lang.ops as ops_mod
from taichi.lang.expr import Expr
from taichi.lang.impl import static
from taichi.lang.kernel_impl import func, pyfunc
from taichi.lang.matrix import Matrix, Vector
from taichi.lang.matrix_ops_utils import (Or, arg_at, assert_list,
                                          assert_tensor, assert_vector,
                                          check_matmul, dim_lt, foreach,
                                          is_int_const, preconditions,
                                          same_shapes, square_matrix)
from taichi.lang.util import cook_dtype
from taichi.types.annotations import template


def _init_matrix(shape, dt=None):
    @pyfunc
    def init():
        return Matrix([[0 for _ in static(range(shape[1]))]
                       for _ in static(range(shape[0]))],
                      dt=dt)

    return init()


def _init_vector(shape, dt=None):
    @pyfunc
    def init():
        return Vector([0 for _ in static(range(shape[0]))], dt=dt)

    return init()


@preconditions(arg_at(0, assert_tensor))
@pyfunc
def _reduce(mat, fun: template()):
    shape = static(mat.get_shape())
    if static(len(shape) == 1):
        result = mat[0]
        for i in static(range(1, shape[0])):
            result = fun(result, mat[i])
        return result
    result = mat[0, 0]
    for i in static(range(shape[0])):
        for j in static(range(shape[1])):
            if static(i != 0 or j != 0):
                result = fun(result, mat[i, j])
    return result


@preconditions(
    arg_at(
        0,
        foreach(
            Or(assert_vector,
               assert_list,
               msg="Cols/rows must be a list of lists, or a list of vectors")),
        same_shapes))
def rows(rows):  # pylint: disable=W0621
    if isinstance(rows[0], (Matrix, Expr)):
        shape = rows[0].get_shape()
        assert len(shape) == 1, "Rows must be a list of vectors"

        @pyfunc
        def _rows():
            return Matrix([[row[i] for i in range(shape[0])] for row in rows])

        return _rows()
    if isinstance(rows[0], list):

        @pyfunc
        def _rows():
            return Matrix([[x for x in row] for row in rows])

        return _rows()
    # unreachable
    return None


@pyfunc
def cols(cols):  # pylint: disable=W0621
    return rows(cols).transpose()


def E(m, x, y, n):
    @func
    def _E():
        return m[x % n, y % n]

    return _E()


@preconditions(square_matrix,
               dim_lt(0, 5,
                      'Determinant of dimension >= 5 is not supported: {}'))
@func
def determinant(x):
    shape = static(x.get_shape())
    if static(shape[0] == 1 and shape[1] == 1):
        return x[0, 0]
    if static(shape[0] == 2 and shape[1] == 2):
        return x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]
    if static(shape[0] == 3 and shape[1] == 3):
        return x[0, 0] * (x[1, 1] * x[2, 2] - x[2, 1] * x[1, 2]) - x[1, 0] * (
            x[0, 1] * x[2, 2] - x[2, 1] * x[0, 2]) + x[2, 0] * (
                x[0, 1] * x[1, 2] - x[1, 1] * x[0, 2])
    if static(shape[0] == 4 and shape[1] == 4):

        det = 0.0
        for i in static(range(4)):
            det += (-1.0)**i * (
                x[i, 0] *
                (E(x, i + 1, 1, 4) *
                 (E(x, i + 2, 2, 4) * E(x, i + 3, 3, 4) -
                  E(x, i + 3, 2, 4) * E(x, i + 2, 3, 4)) - E(x, i + 2, 1, 4) *
                 (E(x, i + 1, 2, 4) * E(x, i + 3, 3, 4) -
                  E(x, i + 3, 2, 4) * E(x, i + 1, 3, 4)) + E(x, i + 3, 1, 4) *
                 (E(x, i + 1, 2, 4) * E(x, i + 2, 3, 4) -
                  E(x, i + 2, 2, 4) * E(x, i + 1, 3, 4))))
        return det
    # unreachable
    return None


def _vector_transpose(v):
    shape = v.get_shape()
    if isinstance(v, Vector):

        @pyfunc
        def _transpose():
            return Matrix([[v[i] for i in static(range(shape[0]))]], ndim=1)

        return _transpose()
    if isinstance(v, Matrix):

        @pyfunc
        def _transpose():
            return Vector([v[i, 0] for i in static(range(shape[0]))])

        return _transpose()
    return v


@preconditions(assert_tensor)
@func
def transpose(m):
    shape = static(m.get_shape())
    if static(len(shape) == 1):
        return _vector_transpose(m)
    result = _init_matrix((shape[1], shape[0]), dt=m.element_type())
    for i in static(range(shape[0])):
        for j in static(range(shape[1])):
            result[j, i] = m[i, j]
    return result


@preconditions(arg_at(0, is_int_const),
               arg_at(
                   1, lambda val:
                   (isinstance(val,
                               (numbers.Number, )) or isinstance(val, Expr),
                    f'Invalid argument type for values: {type(val)}')))
def diag(dim, val):
    dt = val.element_type() if isinstance(val, Expr) else cook_dtype(type(val))

    @func
    def diag_impl():
        result = _init_matrix((dim, dim), dt)
        for i in static(range(dim)):
            result[i, i] = val
        return result

    return diag_impl()


@preconditions(assert_tensor)
@pyfunc
def sum(mat):  # pylint: disable=W0622
    return _reduce(mat, ops_mod.add)


@preconditions(assert_tensor)
@pyfunc
def norm_sqr(mat):
    return sum(mat * mat)


@preconditions(arg_at(0, assert_tensor))
@pyfunc
def norm(mat, eps=0.0):
    return ops_mod.sqrt(norm_sqr(mat) + eps)


@preconditions(arg_at(0, assert_tensor))
@pyfunc
def norm_inv(mat, eps=0.0):
    return ops_mod.rsqrt(norm_sqr(mat) + eps)


@preconditions(arg_at(0, assert_vector))
@pyfunc
def normalized(vec, eps=0.0):
    invlen = 1 / (norm(vec) + eps)
    return invlen * vec


@preconditions(assert_tensor)
@pyfunc
def any(mat):  # pylint: disable=W0622
    return _reduce(mat != 0, ops_mod.bit_or) & True


@preconditions(assert_tensor)
@pyfunc
def all(mat):  # pylint: disable=W0622
    return _reduce(mat != 0, ops_mod.bit_and) & True


@preconditions(assert_tensor)
@pyfunc
def max(mat):  # pylint: disable=W0622
    return _reduce(mat, ops_mod.max_impl)


@preconditions(assert_tensor)
@pyfunc
def min(mat):  # pylint: disable=W0622
    return _reduce(mat, ops_mod.min_impl)


@preconditions(square_matrix)
@pyfunc
def trace(mat):
    shape = static(mat.get_shape())
    result = mat[0, 0]
    # TODO: get rid of static when
    # CHI IR Tensor repr is ready stable
    for i in static(range(1, shape[0])):
        result += mat[i, i]
    return result


@preconditions(arg_at(0, assert_tensor))
@func
def fill(mat: template(), val):
    shape = static(mat.get_shape())
    if static(len(shape) == 1):
        for i in static(range(shape[0])):
            mat[i] = val
        return mat
    for i in static(range(shape[0])):
        for j in static(range(shape[1])):
            mat[i, j] = val
    return mat


@preconditions(check_matmul)
@func
def _matmul_helper(x, y):
    shape_x = static(x.get_shape())
    shape_y = static(y.get_shape())
    if static(len(shape_x) == 1 and len(shape_y) == 1):
        # TODO: Type comparison
        result = _init_matrix((shape_x[0], shape_y[0]), x.element_type())
        for i in static(range(shape_x[0])):
            for j in static(range(shape_y[0])):
                result[i, j] = x[i] * y[j]
        return result
    if static(len(shape_y) == 1):
        # TODO: Type comparison
        result = _init_vector(shape_x, x.element_type())
        for i in static(range(shape_x[0])):
            for j in static(range(shape_x[1])):
                result[i] += x[i, j] * y[j]
        return result
    # TODO: Type comparison
    result = _init_matrix((shape_x[0], shape_y[1]), x.element_type())
    for i in static(range(shape_x[0])):
        for j in static(range(shape_y[1])):
            for k in static(range(shape_x[1])):
                result[i, j] += x[i, k] * y[k, j]
    return result


@func
def matmul(x, y):
    shape_x = static(x.get_shape())
    shape_y = static(y.get_shape())
    if static(len(shape_x) == 1 and len(shape_y) == 2):
        return _matmul_helper(transpose(y), x)
    return _matmul_helper(x, y)
