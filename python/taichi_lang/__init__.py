from .impl import *

print = tprint
core = taichi_lang_core
var = global_var

i = indices(0)
j = indices(1)
k = indices(2)
ij = indices(0, 1)
ijk = indices(0, 1, 2)

__all__ = [kernel, layout, var, global_var, f32, float32, i32, int32, print,
           core, index, make_expr_group, i, j, k, ij, ijk, inside_kernel]
