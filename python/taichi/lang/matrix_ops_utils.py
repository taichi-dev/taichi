import functools

from taichi.lang.exception import TaichiCompilationError
from taichi.lang.expr import Expr
from taichi.lang.matrix import Matrix


def do_check(checker_fns, *args, **kwargs):
    for f in checker_fns:
        try:
            ok, msg = f(*args, **kwargs)
        except TaichiCompilationError as e:
            raise
        if not ok:
            raise TaichiCompilationError(msg)


def preconditions(*checker_funcs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            do_check(checker_funcs, *args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def arg_at(i, *fns):
    def check(*args, **kwargs):
        if i in kwargs:
            arg = kwargs[i]
        else:
            try:
                arg = args[i]
            except IndexError:
                raise
        do_check(fns, arg, **kwargs)
        return True, None

    return check


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
