import taichi as ti


def test_binding():
    taichi_lang = ti.core
    print(taichi_lang.BinaryOpType.mul)
    one = taichi_lang.make_const_expr_i32(1)
    two = taichi_lang.make_const_expr_i32(2)
    expr = taichi_lang.make_binary_op_expr(taichi_lang.BinaryOpType.add, one,
                                           two)
    print(expr.serialize())
    print(taichi_lang.make_global_store_stmt(None, None))
