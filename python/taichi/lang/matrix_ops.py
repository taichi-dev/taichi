from taichi.lang.impl import static
from taichi.lang.kernel_impl import func, pyfunc
from taichi.lang.matrix_ops_utils import (arg_at, assert_tensor, preconditions,
                                          square_matrix)
from taichi.lang.ops import cast
from taichi.lang.util import in_taichi_scope
from taichi.types.annotations import template


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
