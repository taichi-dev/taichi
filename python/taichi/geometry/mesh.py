from taichi.core import tc_core
from taichi.util import *


def t(u, v):
    print 1
    return Vector(u * v, 0, 1)

print t
gen = tc_core.surface_generator_from_py_obj(t)
print gen
v = Vectori(3, 3)
tc_core.generate_mesh(gen, v)
