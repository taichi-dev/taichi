import taichi as ti
import numpy as np

ti.init(arch=ti.cc, log_level=ti.DEBUG)

x = ti.var(ti.f32, ())

print(x[None])
