import taichi as ti
ti.init(ti.opengl, log_level=ti.DEBUG)

a = ti.var(ti.f32, ())

@ti.kernel
def func():
    x = 0
    for i in range(10):
        ti.atomic_add(x, i)
    print(x)

func()
ti.sync()
