import taichi as ti

from taichi.lang.impl import static
from taichi.lang.matrix import Matrix, Vector


def _init_matrix(shape, dt=None):
    return Matrix([[.0 for _ in static(range(shape[1]))] for _ in static(range(shape[0]))], dt=dt)


def _init_vector(shape, dt=None):
    return Vector([.0 for _ in range(shape[0])], dt=dt)

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
    else:
        result = Matrix([[0 for _ in range(shape_y[1])] for _ in range(shape_x[0])], dt=x.element_type())
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
    else:
        return _matmul_helper(x, y)
    
__all__ = ['transpose', 'matmul']