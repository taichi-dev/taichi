import taichi as ti
import numpy as np

ti.init(arch=ti.cc, debug=True, log_level=ti.DEBUG)

res = 512, 512
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
