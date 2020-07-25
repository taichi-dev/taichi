import taichi as ti

x = ti.Vector([2, 3])

x.transposed(x)


@ti.kernel
def func():
    x = 0
    x = 0.1


func()
