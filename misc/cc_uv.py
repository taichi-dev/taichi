import taichi as ti
import numpy as np

ti.init(arch=ti.cc, debug=True, log_level=ti.DEBUG)

res = 512, 512
pixels = ti.Vector(3, dt=ti.f32, shape=res)


@ti.kernel
def paint(t: ti.f32):
    for i, j in pixels:
        u = i / res[0]
        v = j / res[1]
        c = ti.cos(ti.Vector([u, v + 2, u + 4]) + t) * 0.5 + 0.5
        pixels[i, j] = c


gui = ti.GUI('UV', res)
while not gui.get_event(ti.GUI.ESCAPE):
    paint(gui.frame / 60)
    gui.set_image(pixels)
    gui.show()
