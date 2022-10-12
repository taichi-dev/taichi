from taichi.lang.impl import static, subscript
from taichi.lang.matrix import Matrix, Vector
from taichi.lang.matrix_ops_utils import check_det, check_matmul, preconditions

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


@ti.func
def trace(x):
    shape = static(x.get_shape())
    # assert shape[0] == shape[1]
    result = 0
    for i in range(shape[0]):
        result += x[i, i]
    return result


def E(m, x, y, n):
    return subscript(m, x % n, y % n)


@preconditions(check_det)
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


# @ti.func
# def inverse(x):
#     shape = static(x.get_shape())
#     if shape[0] == 1:
#         return Matrix([1 / x[0, 0]])
#     if shape[1] == 2:
#         inv_determinant = impl.expr_init(1.0 / self.determinant())
#         return inv_determinant * Matrix([[self(
#             1, 1), -self(0, 1)], [-self(1, 0), self(0, 0)]])
#     if self.n == 3:
#         n = 3
#         inv_determinant = impl.expr_init(1.0 / self.determinant())
#         entries = [[0] * n for _ in range(n)]

#         def E(x, y):
#             return self(x % n, y % n)

#         for i in range(n):
#             for j in range(n):
#                 entries[j][i] = inv_determinant * (
#                     E(i + 1, j + 1) * E(i + 2, j + 2) -
#                     E(i + 2, j + 1) * E(i + 1, j + 2))
#         return Matrix(entries)
#     if self.n == 4:
#         n = 4
#         inv_determinant = impl.expr_init(1.0 / self.determinant())
#         entries = [[0] * n for _ in range(n)]

#         def E(x, y):
#             return self(x % n, y % n)

#         for i in range(n):
#             for j in range(n):
#                 entries[j][i] = inv_determinant * (-1)**(i + j) * ((
#                     E(i + 1, j + 1) *
#                     (E(i + 2, j + 2) * E(i + 3, j + 3) -
#                         E(i + 3, j + 2) * E(i + 2, j + 3)) - E(i + 2, j + 1) *
#                     (E(i + 1, j + 2) * E(i + 3, j + 3) -
#                         E(i + 3, j + 2) * E(i + 1, j + 3)) + E(i + 3, j + 1) *
#                     (E(i + 1, j + 2) * E(i + 2, j + 3) -
#                         E(i + 2, j + 2) * E(i + 1, j + 3))))
#         return Matrix(entries)
#     raise Exception(
#         "Inversions of matrices with sizes >= 5 are not supported")

__all__ = ['transpose', 'matmul', 'determinant', 'trace']
