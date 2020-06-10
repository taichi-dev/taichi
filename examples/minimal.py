import taichi as ti

ti.init()


@ti.kernel
def p():
    print(42)


p()
