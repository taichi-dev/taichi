import math

import taichi as ti

ti.init(arch=ti.cuda)

n = 256
x = ti.field(ti.f32)
# `bitmasked` is a field that supports sparsity, in that each element can be
# activated individually. (It can be viewed as `dense`, with an extra bit for each
# element to mark its activation). Assigning to an element will activate it
# automatically. Use struct-for syntax to loop over the active elements only.
ti.root.bitmasked(ti.ij, (n, n)).place(x)


@ti.kernel
def activate():
    # All elements in bitmasked is initially deactivated
    # Let's activate elements in the rectangle now!
    for i, j in ti.ndrange((100, 125), (100, 125)):
        x[i, j] = 233  # assign any value to activate the element at (i, j)


@ti.kernel
def paint_active_pixels(color: ti.f32):
    # struct-for syntax: loop over active pixels, inactive pixels are skipped
    for i, j in x:
        x[i, j] = color


@ti.kernel
def paint_all_pixels(color: ti.f32):
    # range-for syntax: loop over all pixels, no matter active or not
    for i, j in ti.ndrange(n, n):
        x[i, j] = color


def main():
    ti.deactivate_all_snodes()
    activate()

    gui = ti.GUI("bitmasked", (n, n))
    for frame in range(10000):
        color = math.sin(frame * 0.05) * 0.5 + 0.5
        paint_active_pixels(color)
        # paint_all_pixels(color)  # try this and compare the difference!
        gui.set_image(x)
        gui.show()


if __name__ == "__main__":
    main()
