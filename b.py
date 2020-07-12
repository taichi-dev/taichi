import taichi as ti


@ti.kernel
def func():
    m = ti.Vector([1, 2, 3, 0])
    r = ti.select(m, 1, 0)
    print(r)


func()
