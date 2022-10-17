from taichi.lang.impl import static
from taichi.lang.kernel_impl import func, pyfunc
from taichi.lang.matrix_ops_utils import (arg_at, assert_tensor, preconditions,
                                          square_matrix)
from taichi.lang.ops import cast
from taichi.lang.util import in_taichi_scope
from taichi.types.annotations import Template


@preconditions(square_matrix)
@pyfunc
def trace(x):
    shape = static(x.get_shape())
    result = cast(0, x.element_type()) if static(in_taichi_scope()) else 0
    # TODO: get rid of static when
    # CHI IR Tensor repr is ready stable
    for i in static(range(shape[0])):
        result += x[i, i]
    return result


@preconditions(arg_at(0, assert_tensor))
@func
def fill(m: Template(), val):
    s = static(m.get_shape())
    if static(len(s) == 1):
        for i in static(range(s[0])):
            m[i] = val
        return m
    for i in static(range(s[0])):
        for j in static(range(s[1])):
            m[i, j] = val
    return m


__all__ = [
    'trace',
    'fill',
]
