from taichi.lang.impl import static
from taichi.lang.kernel_impl import func
from taichi.lang.matrix_ops_utils import (is_tensor, preconditions,
                                          square_matrix)
from taichi.lang.misc import loop_config
from taichi.lang.util import taichi_scope


@preconditions(square_matrix)
@func
def trace(x):
    shape = static(x.get_shape())
    result = 0
    # TODO: fix parallelization
    loop_config(serialize=True)
    for i in range(shape[0]):
        result += x[i, i]
    return result


@preconditions(lambda m, _: is_tensor(m))
@taichi_scope
def fill(m, val):
    @func
    def fill_impl():
        s = static(m.get_shape())
        if static(len(s) == 1):
            # TODO: fix parallelization
            loop_config(serialize=True)
            for i in range(s[0]):
                m[i] = val
            return
        # TODO: fix parallelization
        loop_config(serialize=True)
        for i in range(s[0]):
            for j in range(s[1]):
                m[i, j] = val

    return fill_impl()


__all__ = [
    'trace',
    'fill',
]
