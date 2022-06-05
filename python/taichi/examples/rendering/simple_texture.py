import taichi as ti

ti.init()

res = (512, 512)
pixels = ti.Vector.field(3, dtype=float, shape=res)


@ti.kernel
def paint():
    for i, j in pixels:
        u = i / res[0]
        v = j / res[1]
        c = ti.sample_texture(ti.Vector([u, v]))
        pixels[i, j] = [c, c, c]


gui = ti.GUI('UV', res)
while not gui.get_event(ti.GUI.ESCAPE):
    paint()
    gui.set_image(pixels)
    gui.show()
