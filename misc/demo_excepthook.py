import taichi as ti
ti.init()
ti.enable_excepthook()


@ti.func
def fun():
    x = ti.Vector([2, 3])
    for i in range(3):
        print(x[i])


@ti.kernel
def kern():
    fun()


kern()
