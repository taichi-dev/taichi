from taichi.lang.impl import static
from taichi.lang.kernel_impl import func, pyfunc
from taichi.lang.matrix_ops_utils import (arg_at, assert_tensor, preconditions,
                                          square_matrix)
from taichi.lang.ops import cast
from taichi.lang.util import in_taichi_scope, taichi_scope
from taichi.types.annotations import template
from taichi.lang.expr import Expr
from taichi.lang.matrix import Matrix, Vector


@taichi_scope
def _init_matrix(shape, dt=None):
    @func
    def init():
        return Matrix([[0 for _ in static(range(shape[1]))]
                       for _ in static(range(shape[0]))],
                      dt=dt)

    return init()


@taichi_scope
def _init_vector(shape, dt=None):
    @func
    def init():
        return Vector([0 for _ in static(range(shape[0]))], dt=dt)

    return init()


@taichi_scope
def matrix_reduce(m, f, init, inplace=False):
    shape = m.get_shape()

    @func
    def _reduce():
        result = init
        for i in static(range(shape[0])):
            for j in static(range(shape[1])):
                if static(inplace):
                    f(result, m[i, j])
                else:
                    result = f(result, m[i, j])
        return result

    return _reduce()


@taichi_scope
def vector_reduce(v, f, init, inplace=False):
    shape = v.get_shape()

    @func
    def _reduce():
        result = init
        for i in static(range(shape[0])):
            if static(inplace):
                f(result, v[i])
            else:
                result = f(result, v[i])
        return result

    return _reduce()


@taichi_scope
def reduce(x, f, init, inplace=False):
    if len(x.get_shape()) == 1:
        return vector_reduce(x, f, init, inplace)
    return matrix_reduce(x, f, init, inplace)


@preconditions(square_matrix)
@func
def trace(x):
    shape = static(x.get_shape())
    result = cast(0, x.element_type())
    # TODO: fix parallelization
    loop_config(serialize=True)
    # TODO: get rid of static when
    # CHI IR Tensor repr is ready stable
    for i in static(range(shape[0])):
        result += x[i, i]
    return result


def E(m, x, y, n):
    return subscript(m, x % n, y % n)


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
        n = 4

        det = 0.0
        # TODO: fix parallelization
        loop_config(serialize=True)
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


@preconditions(arg_at(0, is_tensor))
@taichi_scope
def fill(m, val):
    # capture reference to m
    @func
    def fill_impl():
        s = static(m.get_shape())
        if static(len(s) == 1):
            # TODO: fix parallelization
            loop_config(serialize=True)
            for i in static(range(s[0])):
                m[i] = val
            return m
        # TODO: fix parallelization
        loop_config(serialize=True)
        for i in static(range(s[0])):
            for j in static(range(s[1])):
                m[i, j] = val
        return m

    return fill_impl()


@preconditions(is_tensor)
@func
def transpose(m):
    shape = static(m.get_shape())
    result = _init_matrix(shape, dt=m.element_type())
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
def sum(m):  # pylint: disable=W0622
    # pylint: disable=W0108
    f = lambda x, y: ops_mod.atomic_add(x, y)

    @func
    def sum_impl():
        return reduce(m, f, cast(0, m.element_type()), inplace=True)

    return sum_impl()


@preconditions(assert_tensor)
@func
def norm_sqr(m):
    return sum(m * m)


@preconditions(arg_at(0, assert_tensor))
@func
def norm(m, eps=1e-6):
    return ops_mod.sqrt(norm_sqr(m) + eps)


@preconditions(arg_at(0, assert_tensor))
@func
def norm_inv(m, eps=1e-6):
    return ops_mod.rsqrt(norm_sqr(m) + eps)


@preconditions(assert_tensor)
@taichi_scope
def any(x):  # pylint: disable=W0622
    cmp_fn = lambda r, e: ops_mod.atomic_or(r, ops_mod.cmp_ne(e, 0))

    @func
    def any_impl():
        return 1 & reduce(x, cmp_fn, 0, inplace=True)

    return any_impl()


@preconditions(assert_tensor)
def all(x):  # pylint: disable=W0622

    cmp_fn = lambda r, e: ops_mod.atomic_and(r, ops_mod.cmp_ne(e, 0))

    @func
    def all_impl():
        return reduce(x, cmp_fn, 1, inplace=True)

    return all_impl()


@preconditions(assert_tensor)
def max(x):  # pylint: disable=W0622
    return ops_mod.max(x)


@preconditions(square_matrix)
@pyfunc
def trace(mat):
    shape = static(mat.get_shape())
    result = cast(0, mat.element_type()) if static(in_taichi_scope()) else 0
    # TODO: get rid of static when
    # CHI IR Tensor repr is ready stable
    for i in static(range(shape[0])):
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


__all__ = [
    'trace',
    'fill',
]
