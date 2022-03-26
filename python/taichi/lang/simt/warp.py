from taichi._lib import core as _ti_core
from taichi.lang import expr


def all_nonzero():
    # TODO
    pass


def any_nonzero():
    # TODO
    pass


def unique():
    # TODO
    pass


def ballot():
    # TODO
    pass


def shfl_i32(mask, val, offset):
    # TODO
    pass


def shfl_down_i32(mask, val, offset):
    return expr.Expr(
        _ti_core.insert_internal_func_call(
            "cuda_shfl_down_sync_i32",
            expr.make_expr_group(mask, val, offset, 31), False))


def shfl_up_i32(mask, val, offset):
    # TODO
    pass


def shfl_xor_i32(mask, val, offset):
    # TODO
    pass


def match_any():
    # TODO
    pass


def match_all():
    # TODO
    pass


def active_mask():
    # TODO
    pass


def sync():
    # TODO
    pass


__all__ = [
    'all_nonzero',
    'any_nonzero',
    'unique',
    'ballot',
    'shfl_i32',
    'shfl_up_i32',
    'shfl_down_i32',
    'match_any',
    'match_all',
    'active_mask',
    'sync',
]
