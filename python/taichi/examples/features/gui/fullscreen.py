import taichi as ti

ti.init(ti.gpu)

res = (1920, 1080)
img = ti.Vector.field(3, float, res)


@ti.kernel
def render(t: float):
    for i, j in img:
        a = ti.Vector([i / res[0], j / res[1] + 2, i / res[0] + 4])
        img[i, j] = ti.cos(a + t) * 0.5 + 0.5


gui = ti.GUI("UV", res, fullscreen=True, fast_gui=True)
while not gui.get_event(ti.GUI.ESCAPE):
    render(gui.frame * 0.04)
    gui.set_image(img)
    gui.show()
