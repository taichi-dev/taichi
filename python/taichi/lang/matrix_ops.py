import numbers

import taichi.lang.ops as ops_mod
from taichi.lang.expr import Expr
from taichi.lang.impl import static
from taichi.lang.kernel_impl import func, pyfunc
from taichi.lang.matrix import Matrix, Vector
from taichi.lang.matrix_ops_utils import (arg_at, arg_foreach_check,
                                          assert_list, assert_tensor,
                                          assert_vector, check_matmul, dim_lt,
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


@func
def _filled_vector(n: template(), dtype: template(), val: template()):
    return Vector([val for _ in static(range(n))], dtype)


@func
def _filled_matrix(n: template(), m: template(), dtype: template(),
                   val: template()):
    return Matrix([[val for _ in static(range(m))] for _ in static(range(n))],
                  dtype)


@func
def _unit_vector(n: template(), i: template(), dtype: template()):
    return Vector([i == j for j in static(range(n))], dtype)


@func
def _identity_matrix(n: template(), dtype: template()):
    return Matrix([[i == j for j in static(range(n))]
                   for i in static(range(n))], dtype)


@pyfunc
def _rotation2d_matrix(alpha):
    return Matrix([[ops_mod.cos(alpha), -ops_mod.sin(alpha)],
                   [ops_mod.sin(alpha), ops_mod.cos(alpha)]])


@preconditions(
    arg_at(0, lambda xs: same_shapes(*xs)),
    arg_foreach_check(
        0,
        fns=[assert_vector(), assert_list],
        logic='or',
        msg="Cols/rows must be a list of lists, or a list of vectors"))
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


@pyfunc
def E(mat: template(), x: template(), y: template(), n: template()):
    return mat[x % n, y % n]


@preconditions(square_matrix, dim_lt(0, 5))
@pyfunc
def determinant(mat):
    shape = static(mat.get_shape())
    if static(shape[0] == 1):
        return mat[0, 0]
    if static(shape[0] == 2):
        return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    if static(shape[0] == 3):
        return mat[0, 0] * (
            mat[1, 1] * mat[2, 2] - mat[2, 1] * mat[1, 2]) - mat[1, 0] * (
                mat[0, 1] * mat[2, 2] - mat[2, 1] * mat[0, 2]) + mat[2, 0] * (
                    mat[0, 1] * mat[1, 2] - mat[1, 1] * mat[0, 2])
    if static(shape[0] == 4):
        det = mat[0, 0] * 0  # keep type
        for i in static(range(4)):
            det += (-1)**i * (mat[i, 0] *
                              (E(mat, i + 1, 1, 4) *
                               (E(mat, i + 2, 2, 4) * E(mat, i + 3, 3, 4) -
                                E(mat, i + 3, 2, 4) * E(mat, i + 2, 3, 4)) -
                               E(mat, i + 2, 1, 4) *
                               (E(mat, i + 1, 2, 4) * E(mat, i + 3, 3, 4) -
                                E(mat, i + 3, 2, 4) * E(mat, i + 1, 3, 4)) +
                               E(mat, i + 3, 1, 4) *
                               (E(mat, i + 1, 2, 4) * E(mat, i + 2, 3, 4) -
                                E(mat, i + 2, 2, 4) * E(mat, i + 1, 3, 4))))
        return det
    # unreachable
    return None


@preconditions(square_matrix, dim_lt(0, 5))
@pyfunc
def inverse(mat):
    shape = static(mat.get_shape())
    if static(shape[0] == 1):
        return Matrix([[1.0 / mat[0, 0]]])
    inv_determinant = 1.0 / determinant(mat)
    if static(shape[0] == 2):
        return inv_determinant * Matrix([[mat[1, 1], -mat[0, 1]],
                                         [-mat[1, 0], mat[0, 0]]])
    if static(shape[0] == 3):
        return inv_determinant * Matrix([[
            E(mat, i + 1, j + 1, 3) * E(mat, i + 2, j + 2, 3) -
            E(mat, i + 2, j + 1, 3) * E(mat, i + 1, j + 2, 3)
            for i in static(range(3))
        ] for j in static(range(3))])
    if static(shape[0] == 4):
        return inv_determinant * Matrix([[(-1)**(i + j) * (
            (E(mat, i + 1, j + 1, 4) *
             (E(mat, i + 2, j + 2, 4) * E(mat, i + 3, j + 3, 4) -
              E(mat, i + 3, j + 2, 4) * E(mat, i + 2, j + 3, 4)) -
             E(mat, i + 2, j + 1, 4) *
             (E(mat, i + 1, j + 2, 4) * E(mat, i + 3, j + 3, 4) -
              E(mat, i + 3, j + 2, 4) * E(mat, i + 1, j + 3, 4)) +
             E(mat, i + 3, j + 1, 4) *
             (E(mat, i + 1, j + 2, 4) * E(mat, i + 2, j + 3, 4) -
              E(mat, i + 2, j + 2, 4) * E(mat, i + 1, j + 3, 4))))
                                          for i in static(range(4))]
                                         for j in static(range(4))])
    # unreachable
    return None


@preconditions(assert_tensor)
@pyfunc
def transpose(mat):
    shape = static(mat.get_shape())
    if static(len(shape) == 1):
        return Vector([mat[i] for i in static(range(shape[0]))])
    return Matrix([[mat[i, j] for i in static(range(shape[0]))]
                   for j in static(range(shape[1]))])


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


@preconditions(arg_at(0, assert_vector()))
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


@preconditions(arg_at(0, assert_vector("lhs for dot is not a vector")),
               arg_at(1, assert_vector("rhs for dot is not a vector")))
@pyfunc
def dot(vec_x, vec_y):
    return sum(vec_x * vec_y)


@preconditions(arg_at(0, assert_vector("lhs for cross is not a vector")),
               arg_at(1, assert_vector("rhs for cross is not a vector")),
               same_shapes, arg_at(0, dim_lt(0, 4)))
@pyfunc
def cross(vec_x, vec_y):
    shape = static(vec_x.get_shape())
    if static(shape[0] == 2):
        return vec_x[0] * vec_y[1] - vec_x[1] * vec_y[0]
    if static(shape[0] == 3):
        return Vector([
            vec_x[1] * vec_y[2] - vec_x[2] * vec_y[1],
            vec_x[2] * vec_y[0] - vec_x[0] * vec_y[2],
            vec_x[0] * vec_y[1] - vec_x[1] * vec_y[0]
        ])
    return None


@preconditions(
    arg_at(0, assert_vector("lhs for outer_product is not a vector")),
    arg_at(1, assert_vector("rhs for outer_product is not a vector")))
@pyfunc
def outer_product(vec_x, vec_y):
    shape_x = static(vec_x.get_shape())
    shape_y = static(vec_y.get_shape())
    return Matrix([[vec_x[i] * vec_y[j] for j in static(range(shape_y[0]))]
                   for i in static(range(shape_x[0]))])


@preconditions(assert_tensor)
@func
def cast(mat, dtype: template()):
    return ops_mod.cast(mat, dtype)
