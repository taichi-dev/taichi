import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.exception import TaichiCompilationError, TaichiTypeError
from taichi.lang.matrix import make_matrix
from taichi.lang.util import is_matrix_class, is_taichi_class, to_numpy_type
from taichi.types import primitive_types
from taichi.types.primitive_types import integer_types, real_types


# Scalar, basic data type
class Expr(TaichiOperations):
    """A Python-side Expr wrapper, whose member variable `ptr` is an instance of C++ Expr class. A C++ Expr object contains member variable `expr` which holds an instance of C++ Expression class."""

    def __init__(self, *args, dbg_info=None, dtype=None):
        self.dbg_info = dbg_info
        self.ptr_type_checked = False
        if len(args) == 1:
            if isinstance(args[0], _ti_core.Expr):
                self.ptr = args[0]
            elif isinstance(args[0], Expr):
                self.ptr = args[0].ptr
                self.ptr_type_checked = args[0].ptr_type_checked
                self.dbg_info = args[0].dbg_info
            elif is_matrix_class(args[0]):
                self.ptr = make_matrix(args[0].to_list()).ptr
            elif isinstance(args[0], (list, tuple)):
                self.ptr = make_matrix(args[0]).ptr
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
        if self.dbg_info:
            self.ptr.set_dbg_info(self.dbg_info)
        if not self.ptr_type_checked:
            self.ptr.type_check(impl.get_runtime().prog.config())
            self.ptr_type_checked = True

    def is_tensor(self):
        return self.ptr.is_tensor()

    def is_struct(self):
        return self.ptr.is_struct()

    def element_type(self):
        return self.ptr.get_rvalue_type().element_type()

    def get_shape(self):
        if not self.is_tensor():
            raise TaichiCompilationError(f"Getting shape of non-tensor type: {self.ptr.get_rvalue_type()}")
        return tuple(self.ptr.get_shape())

    @property
    def n(self):
        shape = self.get_shape()
        if len(shape) < 1:
            raise TaichiCompilationError(f"Getting n of tensor type < 1D: {self.ptr.get_rvalue_type()}")
        return shape[0]

    @property
    def m(self):
        shape = self.get_shape()
        if len(shape) < 2:
            raise TaichiCompilationError(f"Getting m of tensor type < 2D: {self.ptr.get_rvalue_type()}")
        return shape[1]

    def __hash__(self):
        return self.ptr.get_raw_address()

    def __str__(self):
        return "<ti.Expr>"

    def __repr__(self):
        return "<ti.Expr>"


def _check_in_range(npty, val):
    iif = np.iinfo(npty)
    return iif.min <= val <= iif.max


def _clamp_unsigned_to_range(npty, val):
    # npty: np.int32 or np.int64
    iif = np.iinfo(npty)
    if iif.min <= val <= iif.max:
        return val
    cap = 1 << iif.bits
    assert 0 <= val < cap
    new_val = val - cap
    return new_val


def make_constant_expr(val, dtype):
    if isinstance(val, (bool, np.bool_)):
        constant_dtype = primitive_types.u1
        return Expr(_ti_core.make_const_expr_bool(constant_dtype, val))

    if isinstance(val, (float, np.floating)):
        constant_dtype = impl.get_runtime().default_fp if dtype is None else dtype
        if constant_dtype not in real_types:
            raise TaichiTypeError(
                "Floating-point literals must be annotated with a floating-point type. For type casting, use `ti.cast`."
            )
        return Expr(_ti_core.make_const_expr_fp(constant_dtype, val))

    if isinstance(val, (int, np.integer)):
        constant_dtype = impl.get_runtime().default_ip if dtype is None else dtype
        if constant_dtype not in integer_types:
            raise TaichiTypeError(
                "Integer literals must be annotated with a integer type. For type casting, use `ti.cast`."
            )
        if _check_in_range(to_numpy_type(constant_dtype), val):
            return Expr(_ti_core.make_const_expr_int(constant_dtype, _clamp_unsigned_to_range(np.int64, val)))
        if dtype is None:
            raise TaichiTypeError(
                f"Integer literal {val} exceeded the range of default_ip: {impl.get_runtime().default_ip}, please specify the dtype via e.g. `ti.u64({val})` or set a different `default_ip` in `ti.init()`"
            )
        else:
            raise TaichiTypeError(f"Integer literal {val} exceeded the range of specified dtype: {dtype}")

    raise TaichiTypeError(f"Invalid constant scalar data type: {type(val)}")


def make_var_list(size, ast_builder=None):
    exprs = []
    for _ in range(size):
        if ast_builder is None:
            exprs.append(impl.get_runtime().prog.make_id_expr(""))
        else:
            exprs.append(ast_builder.make_id_expr(""))
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
        flattened = _get_flattened_ptrs(i)
        for item in flattened:
            expr_group.push_back(item)
    return expr_group


def _get_flattened_ptrs(val):
    if is_taichi_class(val):
        ptrs = []
        for item in val._members:
            ptrs.extend(_get_flattened_ptrs(item))
        return ptrs
    return [Expr(val).ptr]


__all__ = []
