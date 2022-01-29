import taichi as ti


def test_binding():
    ti.init()
    taichi_lang = ti._lib.core
    one = taichi_lang.make_const_expr_i32(1)
    two = taichi_lang.make_const_expr_i32(2)
    grp = taichi_lang.ExprGroup()
    grp.push_back(one)
    grp.push_back(two)
    expr = taichi_lang.call_op(taichi_lang.Operation.add, grp)
    print(expr.serialize())
    print(taichi_lang.make_global_store_stmt(None, None))
