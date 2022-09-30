import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

np_int = np.max([0, 1, 2])
block = ti.root.pointer(ti.i, np_int)
