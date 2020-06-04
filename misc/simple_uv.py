import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, print_ir=True)

res = 1280, 720
pixels = ti.Vector(3, dt=ti.f32, shape=res)


@ti.kernel
def paint(size_x: ti.template(), size_y: ti.template()):
    for i, j in pixels:
        u = i / size_x
        v = j / size_y
        pixels[i, j] = [u, v, 0]


gui = ti.GUI('UV', res)

paint(res[0], res[1])
gui.set_image(pixels)

for i in range(200):
    ti.profiler_start('paint')
    paint(res[0], res[1])
    ti.sync()
    ti.profiler_stop()
    ti.profiler_start('set_image')
    gui.set_image(pixels)
    ti.profiler_stop()
    gui.show()

ti.profiler_print()
