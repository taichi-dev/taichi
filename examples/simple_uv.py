import taichi as ti
import numpy as np

ti.init()

res = 1280, 720
pixels = ti.Vector(3, dt=ti.f32, shape=res)


@ti.kernel
def paint(size_x: ti.template(), size_y: ti.template()):
    for i, j in pixels:
        u = i / size_x
        v = j / size_y
        pixels[i, j] = [u, v, 0]


gui = ti.GUI('UV', res)
while not gui.get_event(ti.GUI.ESCAPE):
    paint(res[0], res[1])
    gui.set_image(pixels)
    gui.show()
