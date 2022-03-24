from taichi._lib import core as _ti_core
from taichi.lang import expr, impl

def shfl_down_sync_i32(mask, val, offset):
    return expr.Expr(_ti_core.insert_internal_func_call("cuda_shfl_down_sync_i32", expr.make_expr_group(mask, val, offset, 31), False))

__all__ = ['shfl_down_sync_i32']