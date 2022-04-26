import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.exception import TaichiTypeError
from taichi.lang.util import is_taichi_class, to_numpy_type, to_taichi_type
from taichi.types.primitive_types import integer_types, real_types


# Scalar, basic data type
class Expr(TaichiOperations):
    """A Python-side Expr wrapper, whose member variable `ptr` is an instance of C++ Expr class. A C++ Expr object contains member variable `expr` which holds an instance of C++ Expression class."""
    def __init__(self, *args, tb=None, dtype=None):
        self.tb = tb
        if len(args) == 1:
            if isinstance(args[0], _ti_core.Expr):
                self.ptr = args[0]
            elif isinstance(args[0], Expr):
                self.ptr = args[0].ptr
                self.tb = args[0].tb
            elif is_taichi_class(args[0]):
                raise TaichiTypeError(
                    'Cannot initialize scalar expression from '
                    f'taichi class: {type(args[0])}')
            else:
                # assume to be constant
                arg = args[0]
                if isinstance(arg, np.ndarray):
                    if arg.shape:
                        raise TaichiTypeError(
                            "Only 0-dimensional numpy array can be used to initialize a scalar expression"
                        )
                    arg = arg.dtype.type(arg)
                self.ptr = make_constant_expr(arg, dtype).ptr
        else:
            assert False
        if self.tb:
            self.ptr.set_tb(self.tb)
        self.ptr.type_check(impl.get_runtime().prog.config)

    def __hash__(self):
        return self.ptr.get_raw_address()

    def __str__(self):
        return '<ti.Expr>'

    def __repr__(self):
        return '<ti.Expr>'


def _check_in_range(npty, val):
    iif = np.iinfo(npty)
    if not iif.min <= val <= iif.max:
        # This isn't the case we want to deal with: |val| does't fall into the valid range of either
        # the signed or the unsigned type.
        raise TaichiTypeError(
            f'Constant {val} has exceeded the range of {to_taichi_type(npty)}: [{iif.min}, {iif.max}]'
        )


def _clamp_unsigned_to_range(npty, val):
    # npty: np.int32 or np.int64
    iif = np.iinfo(npty)
    if iif.min <= val <= iif.max:
        return val
    cap = (1 << iif.bits)
    assert 0 <= val < cap
    new_val = val - cap
    return new_val


def make_constant_expr(val, dtype):
    if isinstance(val, (int, np.integer)):
        constant_dtype = impl.get_runtime(
        ).default_ip if dtype is None else dtype
        if constant_dtype not in integer_types:
            raise TaichiTypeError(
                'Integer literals must be annotated with a integer type. For type casting, use `ti.cast`.'
            )
        _check_in_range(to_numpy_type(constant_dtype), val)
        return Expr(
            _ti_core.make_const_expr_int(
                constant_dtype, _clamp_unsigned_to_range(np.int64, val)))
    if isinstance(val, (float, np.floating)):
        constant_dtype = impl.get_runtime(
        ).default_fp if dtype is None else dtype
        if constant_dtype not in real_types:
            raise TaichiTypeError(
                'Floating-point literals must be annotated with a floating-point type. For type casting, use `ti.cast`.'
            )
        return Expr(_ti_core.make_const_expr_fp(constant_dtype, val))
    raise TaichiTypeError(f'Invalid constant scalar data type: {type(val)}')


def make_var_list(size, ast_builder=None):
    exprs = []
    for _ in range(size):
        if ast_builder is None:
            exprs.append(impl.get_runtime().prog.make_id_expr(''))
        else:
            exprs.append(ast_builder.make_id_expr(''))
    return exprs


def make_expr_group(*exprs):
    from taichi.lang.matrix import Matrix  # pylint: disable=C0415
    if len(exprs) == 1:
        if isinstance(exprs[0], (list, tuple)):
            exprs = exprs[0]
        elif isinstance(exprs[0], Matrix):
            mat = exprs[0]
            assert mat.m == 1
            exprs = mat.entries
    expr_group = _ti_core.ExprGroup()
    for i in exprs:
        if isinstance(i, Matrix):
            assert i.local_tensor_proxy is not None
            expr_group.push_back(i.local_tensor_proxy)
        else:
            expr_group.push_back(Expr(i).ptr)
    return expr_group


__all__ = []
