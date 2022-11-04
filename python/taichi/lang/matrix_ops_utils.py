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


def arg_at(i, *fns):
    def check(*args, **kwargs):
        if i in kwargs:
            arg = kwargs[i]
        else:
            try:
                arg = args[i]
            except IndexError:
                raise
        ok, msg = do_check(fns, arg)
        if not ok:
            return False, msg
        return True, None

    return check


def foreach(*fns):
    def check(args):
        for x in args:
            ok, msg = do_check(fns, x)
            if not ok:
                return False, msg
        return True, None

    return check


def Or(f, g, msg=None):
    def check(*args, **kwargs):
        ok, msg_f = do_check([f], *args, **kwargs)
        if not ok:
            ok, msg_g = do_check([g], *args, **kwargs)
            if not ok:
                return False, f'Both violated: {msg_f} {msg_g}'
        return True, None

    return check


def assert_tensor(m, msg='not tensor type: {}'):
    if isinstance(m, Matrix):
        return True, None
    if isinstance(m, Expr) and m.is_tensor():
        return True, None
    raise TaichiCompilationError(msg.format(type(m)))


# TODO(zhanlue): rearrange to more generic checker functions
# for example: "assert_is_instance(args, indices=[], instances=[], logic='or')"
def assert_vector(v, msg='not a vector: {}'):
    if (isinstance(v, Expr) or isinstance(v, Matrix)) and len(
            v.get_shape()) == 1:
        return True, None
    raise TaichiCompilationError(msg.format(type(v)))


def assert_list(x, msg='not a list: {}'):
    if isinstance(x, list):
        return True, None
    raise TaichiCompilationError(msg.format(type(x)))


def same_shapes(xs):
    shapes = [x.get_shape() for x in xs]
    if len(set(shapes)) != 1:
        return False, f'required shapes to be the same, got shapes {shapes}'
    return True, None


def square_matrix(x):
    assert_tensor(x)
    shape = x.get_shape()
    if shape[0] != shape[1]:
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
