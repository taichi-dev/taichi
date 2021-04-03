from taichi.core.util import ti_core as _ti_core
from taichi.lang.expr import Expr
from taichi.lang.snode import SNode
from taichi.lang.util import cook_dtype, to_taichi_type


class ArgExtArray:
    def __init__(self, dim=1):
        assert dim == 1

    def extract(self, x):
        return to_taichi_type(x.dtype), len(x.shape)


ext_arr = ArgExtArray


class Template:
    def __init__(self, tensor=None, dim=None):
        self.tensor = tensor
        self.dim = dim

    def extract(self, x):
        if isinstance(x, SNode):
            return x.ptr
        if isinstance(x, Expr):
            return x.ptr.get_underlying_ptr_address()
        if isinstance(x, _ti_core.Expr):
            return x.get_underlying_ptr_address()
        if isinstance(x, tuple):
            return tuple(self.extract(item) for item in x)
        return x


template = Template


def decl_scalar_arg(dtype):
    dtype = cook_dtype(dtype)
    arg_id = _ti_core.decl_arg(dtype, False)
    return Expr(_ti_core.make_arg_load_expr(arg_id, dtype))


def decl_ext_arr_arg(dtype, dim):
    dtype = cook_dtype(dtype)
    arg_id = _ti_core.decl_arg(dtype, True)
    return Expr(_ti_core.make_external_tensor_expr(dtype, dim, arg_id))


def decl_scalar_ret(dtype):
    dtype = cook_dtype(dtype)
    return _ti_core.decl_ret(dtype)
