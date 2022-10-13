import functools

from taichi.lang.exception import TaichiCompilationError
from taichi.lang.expr import Expr
from taichi.lang.matrix import Matrix


def preconditions(*checker_funcs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for f in checker_funcs:
                try:
                    ok, msg = f(*args, **kwargs)
                except TaichiCompilationError as _:
                    raise
                if not ok:
                    raise TaichiCompilationError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def forall(func):
    def check(*args, **_):
        for i, arg in enumerate(args):
            ok, msg = func(arg)
            if not ok:
                raise TaichiCompilationError(
                    f"#{i} argument violates the precondition.\n" + msg)

    return check


def is_tensor(m, msg='not tensor type: {}'):
    if isinstance(m, Matrix):
        return True, None
    if isinstance(m, Expr) and m.is_tensor():
        return True, None
    raise TaichiCompilationError(msg.format(type(m)))


def is_matrix(x):
    is_tensor(x)
    s = x.get_shape()
    return len(s) == 2, f'not a matrix: {s}'


def is_vector(x):
    is_tensor(x)
    s = x.get_shape()
    return len(s) == 1, f'not a vector: {s}'


def check_matmul(x, y):
    is_tensor(x, f'left hand side is not a matrix: {type(x)}')
    is_tensor(y, f'right hand side is not a matrix: {type(y)}')
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    if len(x_shape) == 1:
        if x_shape[0] != y_shape[1]:
            return False, f'dimension mismatch between {x_shape} and {y_shape} for left multiplication'
    else:
        if x_shape[0] != y_shape[0]:
            return False, f'dimension mismatch between {x_shape} and {y_shape} for matrix multiplication'
    return True, None


def square_matrix(x):
    is_tensor(x)
    shape = x.get_shape()
    if shape[0] != shape[1]:
        return False, f'not a square matrix: {shape}'
    return True, None


def dim_lt(dim, limit, msg=None):
    def check(x):
        is_tensor(x)
        shape = x.get_shape()
        return shape[dim] < limit, (
            f'Dimension >= {limit} is not supported: {shape}'
            if not msg else msg.format(shape))

    return check


def is_int_const(x):
    if isinstance(x, int):
        return True, None
    if isinstance(x, Expr) and x.val_int() is not None:
        return True, None
    return False, f'not an integer: {type(x)}'
