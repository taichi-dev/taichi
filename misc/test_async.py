import taichi as ti

ti.init()

n = 8

x = ti.var(dt=ti.i32, shape=n)


@ti.kernel
def fill():
    for i in range(n):
        # x[i] = i
        print(i)


fill()
fill()
