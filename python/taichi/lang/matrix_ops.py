import numbers

from taichi.lang.expr import Expr
from taichi.lang.impl import static, subscript
from taichi.lang.matrix import Matrix, Vector
from taichi.lang.matrix_ops_utils import (check_matmul, dim_lt, is_int_const,
                                          is_tensor, preconditions,
                                          square_matrix)

import taichi as ti


def _init_matrix(shape, dt=None):
    return Matrix([[.0 for _ in static(range(shape[1]))]
                   for _ in static(range(shape[0]))],
                  dt=dt)


def _init_vector(shape, dt=None):
    return Vector([.0 for _ in range(shape[0])], dt=dt)


@preconditions(check_matmul)
@ti.func
def _matmul_helper(x, y):
    shape_x = static(x.get_shape())
    shape_y = static(y.get_shape())
    if static(len(shape_y) == 1):
        result = Vector([0 for _ in range(shape_x[0])])
        # TODO: fix parallelization
        ti.loop_config(serialize=True)
        for i in range(shape_x[0]):
            for j in range(shape_y[1]):
                for k in range(shape_x[1]):
                    result[i] += x[i, k] * y[k, j]
        return result
    result = Matrix([[0 for _ in range(shape_y[1])]
                     for _ in range(shape_x[0])],
                    dt=x.element_type())
    # TODO: fix parallelization
    ti.loop_config(serialize=True)
    for i in range(shape_x[0]):
        for j in range(shape_y[1]):
            for k in range(shape_x[1]):
                result[i, j] += x[i, k] * y[k, j]
    return result


@ti.func
def transpose(x):
    shape = static(x.get_shape())
    result = _init_matrix((shape[1], shape[0]), dt=x.element_type())
    # TODO: fix parallelization
    ti.loop_config(serialize=True)
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[j, i] = x[i, j]
    return result


@ti.func
def matmul(x, y):
    shape_x = static(x.get_shape())
    shape_y = static(y.get_shape())
    if static(len(shape_x) == 1 and len(shape_y) == 2):
        return _matmul_helper(transpose(y), x)
    return _matmul_helper(x, y)


@preconditions(square_matrix)
@ti.func
def trace(x):
    shape = static(x.get_shape())
    result = 0
    for i in range(shape[0]):
        result += x[i, i]
    return result


def E(m, x, y, n):
    return subscript(m, x % n, y % n)


@preconditions(square_matrix,
               dim_lt(0, 5,
                      'Determinant of dimension >= 5 is not supported: {}'))
@ti.func
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
        n = 4

        det = 0.0
        for i in range(4):
            det += (-1.0)**i * (
                x[i, 0] *
                (E(x, i + 1, 1, n) *
                 (E(x, i + 2, 2, n) * E(x, i + 3, 3, n) -
                  E(x, i + 3, 2, n) * E(x, i + 2, 3, n)) - E(x, i + 2, 1, n) *
                 (E(x, i + 1, 2, n) * E(x, i + 3, 3, n) -
                  E(x, i + 3, 2, n) * E(x, i + 1, 3, n)) + E(x, i + 3, 1, n) *
                 (E(x, i + 1, 2, n) * E(x, i + 2, 3, n) -
                  E(x, i + 2, 2, n) * E(x, i + 1, 3, n))))
        return det
    # unreachable
    return None


@preconditions(square_matrix,
               dim_lt(0, 5, 'Inverse of dimension >= 5 is not supported: {}'))
@ti.func
def inverse(x):
    n = static(x.get_shape()[0])
    if static(n == 1):
        return Matrix([1 / x[0, 0]])
    if static(n == 2):
        inv_determinant = 1.0 / determinant(x)
        return inv_determinant * Matrix([[x[1, 1], -x[0, 1]],
                                         [-x[1, 0], x[0, 0]]])
    if static(n == 3):
        n = 3
        inv_determinant = 1.0 / determinant(x)
        result = Matrix([[0] * n for _ in range(n)])

        # TODO: fix parallelization
        ti.loop_config(serialize=True)
        for i in range(n):
            for j in range(n):
                result[j, i] = inv_determinant * (
                    E(x, i + 1, j + 1, n) * E(x, i + 2, j + 2, n) -
                    E(x, i + 2, j + 1, n) * E(x, i + 1, j + 2, n))
        return result
    if static(n == 4):
        n = 4
        inv_determinant = 1.0 / determinant(x)
        result = Matrix([[0] * n for _ in range(n)])

        # TODO: fix parallelization
        ti.loop_config(serialize=True)
        for i in range(n):
            for j in range(n):
                result[j, i] = inv_determinant * (-1)**(i + j) * (
                    (E(x, i + 1, j + 1, n) *
                     (E(x, i + 2, j + 2, n) * E(x, i + 3, j + 3, n) -
                      E(x, i + 3, j + 2, n) * E(x, i + 2, j + 3, n)) -
                     E(x, i + 2, j + 1, n) *
                     (E(x, i + 1, j + 2, n) * E(x, i + 3, j + 3, n) -
                      E(x, i + 3, j + 2, n) * E(x, i + 1, j + 3, n)) +
                     E(x, i + 3, j + 1, n) *
                     (E(x, i + 1, j + 2, n) * E(x, i + 2, j + 3, n) -
                      E(x, i + 2, j + 2, n) * E(x, i + 1, j + 3, n))))
        return result
    return None


@preconditions(is_tensor)
@ti.func
def transpose(m):
    shape = static(m.get_shape())
    if static(len(shape) == 1):
        return m
    result = Matrix([[-1 for _ in range(shape[0])] for _ in range(shape[1])],
                    dt=m.element_type())

    # TODO: fix parallelization
    ti.loop_config(serialize=True)
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[j, i] = m[i, j]
    return result


@preconditions(lambda dim, _: is_int_const(dim), lambda _, val:
               (isinstance(val, (numbers.Number, )) or isinstance(val, Expr),
                f'Invalid argument type: {type(val)}'))
def diag(dim, val):
    dt = val.element_type() if isinstance(val, Expr) else type(val)

    @ti.func
    def diag_impl():
        result = Matrix([[0 for _ in range(dim)] for _ in range(dim)], dt=dt)
        ti.loop_config(serialize=True)
        for i in range(dim):
            result[i, i] = val
        return result

    return diag_impl()


@preconditions(is_tensor)
@ti.func
# pylint: disable=W0622
def sum(m):
    result = ti.cast(0, m.element_type())
    s = static(m.get_shape())
    if static(len(s) == 1):
        for i in range(s[0]):
            result += m[i]
        return result
    for i in range(s[0]):
        for j in range(s[1]):
            result += m[i, j]
    return result


@preconditions(is_tensor)
@ti.func
def norm_sqr(m):
    return sum(m * m)


@preconditions(lambda x, **_: is_tensor(x))
def norm(m, eps=1e-6):
    @ti.func
    def norm_impl():
        return ti.sqrt(norm_sqr(m) + eps)

    return norm_impl()


@preconditions(lambda x, **_: is_tensor(x))
def norm_inv(m, eps=1e-6):
    @ti.func
    def norm_inv_impl():
        return ti.rsqrt(norm_sqr(m) + eps)

    return norm_inv_impl()


__all__ = [
    'transpose', 'matmul', 'determinant', 'trace', 'inverse', 'transpose',
    'diag', 'sum', 'norm_sqr', 'norm', 'norm_inv'
]
