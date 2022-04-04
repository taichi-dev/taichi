from taichi._lib import core as _ti_core
from taichi.lang import expr
from taichi.types import i32


def barrier():
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupBarrier",
                                           expr.make_expr_group(), False))


def memory_barrier():
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupMemoryBarrier",
                                           expr.make_expr_group(), False))


def elect():
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupElect",
                                           expr.make_expr_group(), False))


def all_true(cond):
    # TODO
    pass


def any_true(cond):
    # TODO
    pass


def all_equal(value):
    # TODO
    pass


def broadcast_first(value):
    # TODO
    pass

def broadcast(value, index : i32):
  return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupBroadcast",
                                           expr.make_expr_group(value, index),
                                           False))

def group_size():
    return expr.Expr(_ti_core.insert_internal_func_call(
        "subgroupSize", expr.make_expr_group(), False),
                     dtype=i32)


def invocation_id():
    return expr.Expr(_ti_core.insert_internal_func_call(
        "subgroupInvocationId", expr.make_expr_group(), False),
                     dtype=i32)


def reduce_add(value):
    return expr.Expr(_ti_core.insert_internal_func_call(
        "subgroupAdd", expr.make_expr_group(value), False),
                     dtype=value.dtype)


def reduce_mul(value):
    return expr.Expr(_ti_core.insert_internal_func_call(
        "subgroupMul", expr.make_expr_group(value), False),
                     dtype=value.dtype)


def reduce_min(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupMin",
                                           expr.make_expr_group(value), False))


def reduce_max(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupMax",
                                           expr.make_expr_group(value), False))


def reduce_and(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupAnd",
                                           expr.make_expr_group(value), False))


def reduce_or(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupOr",
                                           expr.make_expr_group(value), False))


def reduce_xor(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupXor",
                                           expr.make_expr_group(value), False))


def inclusive_add(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupInclusiveAdd",
                                           expr.make_expr_group(value), False))


def inclusive_mul(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupInclusiveMul",
                                           expr.make_expr_group(value), False))


def inclusive_min(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupInclusiveMin",
                                           expr.make_expr_group(value), False))


def inclusive_max(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupInclusiveMax",
                                           expr.make_expr_group(value), False))


def inclusive_and(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupInclusiveAnd",
                                           expr.make_expr_group(value), False))


def inclusive_or(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupInclusiveOr",
                                           expr.make_expr_group(value), False))


def inclusive_xor(value):
    return expr.Expr(
        _ti_core.insert_internal_func_call("subgroupInclusiveXor",
                                           expr.make_expr_group(value), False))


def exclusive_add(value):
    # TODO
    pass


def exclusive_mul(value):
    # TODO
    pass


def exclusive_min(value):
    # TODO
    pass


def exclusive_max(value):
    # TODO
    pass


def exclusive_and(value):
    # TODO
    pass


def exclusive_or(value):
    # TODO
    pass


def exclusive_xor(value):
    # TODO
    pass


def shuffle(value, index):
    # TODO
    pass


def shuffle_xor(value, mask):
    # TODO
    pass


def shuffle_up(value, offset):
    # TODO
    pass


def shuffle_down(value, offset):
    # TODO
    pass


__all__ = [
    'barrier', 'memory_barrier', 'elect', 'all_true', 'any_true', 'all_equal',
    'broadcast_first', 'reduce_add', 'reduce_mul', 'reduce_min', 'reduce_max',
    'reduce_and', 'reduce_or', 'reduce_xor', 'inclusive_add', 'inclusive_mul',
    'inclusive_min', 'inclusive_max', 'inclusive_and', 'inclusive_or',
    'inclusive_xor', 'exclusive_add', 'exclusive_mul', 'exclusive_min',
    'exclusive_max', 'exclusive_and', 'exclusive_or', 'exclusive_xor',
    'shuffle', 'shuffle_xor', 'shuffle_up', 'shuffle_down'
]
