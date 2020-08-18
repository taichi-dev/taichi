import taichi as ti

ti.init(ti.opengl)


@ti.kernel
def func():
    print(42)


func()
