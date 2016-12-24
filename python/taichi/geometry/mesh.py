from taichi.core import tc_core
from taichi.util import *

gen = tc_core.surface_generator_from_py_obj(lambda u, v: Vector(u, v, 1.234))
tc_core.generate_mesh(gen, Vectori(10, 10))
