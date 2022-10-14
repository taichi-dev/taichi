from taichi.lang.impl import static
from taichi.lang.kernel_impl import func
from taichi.lang.matrix_ops_utils import (arg_at, is_tensor, preconditions,
                                          square_matrix)
from taichi.lang.misc import loop_config
from taichi.lang.ops import cast
from taichi.lang.util import taichi_scope


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


__all__ = [
    'trace',
    'fill',
]
