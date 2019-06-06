from .impl import *
from .matrix import Matrix

print = tprint
core = taichi_lang_core

i = indices(0)
j = indices(1)
k = indices(2)
ij = indices(0, 1)
ijk = indices(0, 1, 2)
Vector = Matrix

__all__ = [kernel, layout, var, global_var, f64, float64, f32, float32, i32,
           int32, print, core, index, make_expr_group, i, j, k, ij, ijk,
           inside_kernel, Matrix, Vector]
