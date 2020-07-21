import taichi as ti
import numpy as np

ti.init(arch=ti.cc, log_level=ti.DEBUG, debug=True)

tensor = ti.var(ti.i32, 2)


@ti.kernel
def func1():
    for I in tensor:
        tensor[I] = 2
        print('f1', tensor[I])


@ti.kernel
def func2():
    for I in tensor:
        print('f2', tensor[I])


func1()
func2()
