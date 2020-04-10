import taichi as ti

ti.init()

n = 8

x = ti.var(dt=ti.i32, shape=n)


@ti.kernel
def fill():
    for i in x:
        print(x[i])

    for i in x:
        x[i] = i * 2


fill()
fill()

ti.core.print_stat()
