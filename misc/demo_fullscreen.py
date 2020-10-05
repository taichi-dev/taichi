import taichi as ti
import numpy as np

ti.init(ti.gpu)

res = (1920, 1080)


@ti.kernel
def paint(img: ti.ext_arr()):
    for i, j in ti.ndrange(*res):
        u = i / res[0]
        v = j / res[1]
        img[(j * 1920 + i) * 4 + 1] = int(u * 255)
        img[(j * 1920 + i) * 4 + 2] = int(v * 255)
        img[(j * 1920 + i) * 4 + 0] = 0


img = (np.random.rand(1080 * 1920 * 4) * 255).astype(np.uint8)
gui = ti.GUI('UV', res, fullscreen=False, fast_gui=True)
while not gui.get_event(ti.GUI.ESCAPE):
    paint(gui.img)
    gui.show()
