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
                except TaichiCompilationError as e:
                    raise
                if not ok:
                    raise TaichiCompilationError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_tensor(m, msg='not tensor type: {}'):
    if isinstance(m, Matrix):
        return True, None
    if isinstance(m, Expr) and m.is_tensor():
        return True, None
    raise TaichiCompilationError(msg.format(type(m)))


def square_matrix(x):
    is_tensor(x)
    shape = x.get_shape()
    if shape[0] != shape[1]:
        return False, f'not a square matrix: {shape}'
    return True, None
