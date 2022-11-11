import functools

from taichi.lang.exception import TaichiCompilationError
from taichi.lang.expr import Expr
from taichi.lang.matrix import Matrix


def do_check(checker_fns, *args, **kwargs):
    for f in checker_fns:
        ok, msg = f(*args, **kwargs)
        if not ok:
            return False, msg
    return True, None


def preconditions(*checker_funcs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ok, msg = do_check(checker_funcs, *args, **kwargs)
            if not ok:
                raise TaichiCompilationError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def arg_at(indices, *fns):
    def check(*args, **kwargs):
        nonlocal indices
        if isinstance(indices, int):
            indices = [indices]
        for i in indices:
            if i in kwargs:
                arg = kwargs[i]
            else:
                arg = args[i]
            ok, msg = do_check(fns, arg)
            if not ok:
                return False, msg
        return True, None

    return check


def assert_tensor(m, msg='not tensor type: {}'):
    if isinstance(m, Matrix):
        return True, None
    if isinstance(m, Expr) and m.is_tensor():
        return True, None
    return False, msg.format(type(m))


def assert_vector(msg='expected a vector, got {}'):
    def check(v):
        if (isinstance(v, Expr) or isinstance(v, Matrix)) and len(
                v.get_shape()) == 1:
            return True, None
        return False, msg.format(type(v))

    return check


def assert_list(x, msg='not a list: {}'):
    if isinstance(x, list):
        return True, None
    return False, msg.format(type(x))


def arg_foreach_check(*arg_indices, fns=[], logic='or', msg=None):
    def check(*args, **kwargs):
        for i in arg_indices:
            if i in kwargs:
                arg = kwargs[i]
            else:
                arg = args[i]
            if logic == 'or':
                passed = False
                for a in arg:
                    for fn in fns:
                        ok, _ = do_check([fn], a)
                        if ok:
                            passed = True
                            break
                    if not passed:
                        return False, msg
            elif logic == 'and':
                for a in arg:
                    ok, _ = do_check(fns, a)
                    if not ok:
                        return False, msg
            else:
                raise ValueError(f'Unknown logic: {logic}')
        return True, None

    return check


def same_shapes(*xs):
    shapes = [x.get_shape() for x in xs]
    if len(set(shapes)) != 1:
        return False, f'required shapes to be the same, got shapes {shapes}'
    return True, None


def square_matrix(x):
    assert_tensor(x)
    shape = x.get_shape()
    if len(shape) != 2 or shape[0] != shape[1]:
        return False, f'not a square matrix: {shape}'
    return True, None


def dim_lt(dim, limit, msg=None):
    def check(x):
        assert_tensor(x)
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
    return False, f'not an integer: {x} of type {type(x).__name__}'


def check_matmul(x, y):
    assert_tensor(x, f'left hand side is not a matrix: {type(x)}')
    assert_tensor(y, f'right hand side is not a matrix: {type(y)}')
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    if len(x_shape) == 1:
        if len(y_shape) == 1:
            return True, None
        if x_shape[0] != y_shape[0]:
            return False, f'dimension mismatch between {x_shape} and {y_shape} for left multiplication'
    else:
        if x_shape[1] != y_shape[0]:
            return False, f'dimension mismatch between {x_shape} and {y_shape} for matrix multiplication'
    return True, None
