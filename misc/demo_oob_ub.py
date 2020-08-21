# https://forum.taichi.graphics/t/taichi/1003
import taichi as ti

ti.init(arch=ti.cpu)

N = 3

x = ti.field(ti.i32, N)


@ti.kernel
def test():
    for i in x:
        x[i] = 1000 + i
    for i in ti.static(range(-N, 2 * N)):
        print(i, x[i])


test()
