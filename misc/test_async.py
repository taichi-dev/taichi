import taichi as ti

# ti.init(arch=ti.cpu, async_mode=True)
ti.init(arch=ti.cuda, async_mode=True)

n = 8

x = ti.field(dtype=ti.i32, shape=n)


@ti.kernel
def fill():
    for i in x:
        print(x[i])

    for i in x:
        x[i] = i * 2


fill()
fill()

ti.sync()

ti.core.print_stat()
