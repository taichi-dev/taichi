from core import taichi_lang

print(taichi_lang.BinaryOpType.mul)
one = taichi_lang.make_constant_expr(1)
two = taichi_lang.make_constant_expr(2)
expr = taichi_lang.make_binary_op_expr(taichi_lang.BinaryOpType.add, one, two)
print(expr.serialize())

print(taichi_lang.make_global_store_stmt(None, None))