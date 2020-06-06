import taichi as ti


@ti.kernel
def p():
    print(42)


p()
