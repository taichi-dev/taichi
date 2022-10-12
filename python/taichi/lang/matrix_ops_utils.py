from taichi.lang.exception import TaichiCompilationError
from taichi.lang.expr import Expr
from taichi.lang.matrix import Matrix, Vector
import functools

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


def check_matrix(m, msg):
    if isinstance(m, Matrix):
        return True
    if isinstance(m, Expr) and m.is_tensor():
        return True
    raise TaichiCompilationError(msg)
    

def check_matmul(x, y):
    check_matrix(x, f'left hand side is not a matrix: {type(x)}')
    check_matrix(y, f'right hand side is not a matrix: {type(y)}')
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    if len(x_shape) == 1:
        if x_shape[0] != y_shape[1]:
            return False, f'dimension mismatch between {x_shape} and {y_shape} for left multiplication'
    else:
        if x_shape[0] != y_shape[0]:
            return False, f'dimension mismatch between {x_shape} and {y_shape} for matrix multiplication'
    return True


def check_det(x):
    check_matrix(x, f'argument to det(.) is not a matrix: {type(x)}')
    x_shape = x.get_shape()
    if len(x_shape) != 2:
        return False, f'argument to det(.) is not a 2D matrix: {x_shape}'
    if x_shape[0] != x_shape[1]:
        return False, f'argument to det(.) is not a square matrix: {x_shape}'
    if x_shape[0] > 4:
        return False, f'Determinants of matrices with sizes >= 5 are not supported: {x_shape}'