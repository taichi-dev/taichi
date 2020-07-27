import taichi as ti
import numpy as np

ti.init()

res = 1280, 720
pixels = ti.Vector(3, dt=ti.f32, shape=res)


@ti.kernel
def paint():
    for i, j in pixels:
        u = i / res[0]
        v = j / res[1]
        pixels[i, j] = [u, v, 0]


gui = ti.GUI('UV', res)
while not gui.get_event(ti.GUI.ESCAPE):
    paint()
    gui.set_image(pixels)
    gui.show()
