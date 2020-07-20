import taichi as ti
import numpy as np

ti.init(arch=ti.opengl, log_level=ti.DEBUG, debug=True)

@ti.kernel
def func(e: ti.ext_arr()):
    e[0] = 666

e = np.array([233, 233], dtype=np.int32)
func(e)
ti.sync()
print(e)
