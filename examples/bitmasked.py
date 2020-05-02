import taichi as ti
import math

ti.init(arch=ti.gpu)

n = 256
x = ti.var(ti.f32)
# `bitmasked` is a tensor that supports sparsity, in that each element can be
# activated individually. (It can be viewed as `dense`, with an extra bit for each
# element to mark its activation). Assigning to an element will activate it
# automatically. Use struct-for syntax to loop over the active elements only.
ti.root.bitmasked(ti.ij, (n, n)).place(x)


@ti.kernel
def activate():
    # All elements in bitmasked is initially deactivated
    # Let's activate elements in the rectangle now!
    for i, j in ti.ndrange((100, 125), (100, 125)):
        x[i, j] = 0  # assign any value to activate the element at (i, j)


@ti.kernel
def paint_active_pixels(t: ti.f32):
    # struct-for syntax: loop over active pixels, inactive pixels are excluded
    for i, j in x:
        x[i, j] = t


@ti.kernel
def paint_all_pixels(t: ti.f32):
    # range-for syntax: loop over all pixels, no matter active or not
    for i, j in ti.ndrange(n, n):
        x[i, j] = t


gui = ti.GUI('bitmasked', (n, n))
for frame in range(10000):
    ti.root.deactivate_all()
    activate()
    t = math.sin(frame * 0.05) * 0.5 + 0.5
    paint_active_pixels(t)
    #paint_all_pixels(t)  # try this and compare the difference!
    gui.set_image(x)
    gui.show()
