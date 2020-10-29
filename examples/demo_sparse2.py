import taichi as ti

ti.init()

n = 16
x = ti.var(ti.f32)
gui_res = 512
img = ti.field(ti.f32, shape=(gui_res, gui_res))

cell = ti.root.pointer(ti.ij, n)
cell.place(x)

scale = gui_res // n


@ti.kernel
def activate():
    for i, j in ti.ndrange(n, n):
        if i < j:
            x[i, j] = 1

    ti.activate(cell, [10, 3])


@ti.kernel
def deact():
    ti.deactivate(cell, [5, 10])


@ti.kernel
def paint():
    for i, j in img:
        img[i, j] = ti.is_active(cell, [i // scale, j // scale])


gui = ti.GUI('pointer', (gui_res, gui_res))

activate()
deact()
paint()

while True:
    gui.set_image(img.to_numpy())
    gui.show()
