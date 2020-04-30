import taichi as ti

ti.init()

n = 256
x = ti.var(ti.f32)
ti.root.bitmasked(ti.ij, (n, n)).place(x)
# `bitmasked` is a kind of tensor, adds an extra bit to every element in it.
# So every elements in a bitmasked tensor can be either **active or inactive**.
# Assigning to an element will cause that element to become active.
# Use struct-for syntax to loop over active elements only.


@ti.kernel
def activate():
    # All elements in bitmasked is deactivated at initial.
    # Let's activate elements in the rectangle now!
    for i, j in ti.ndrange((100, 125), (100, 125)):  # loop over a rectangle area
        # In taichi, in order to activate an element,
        # simply **assign any value** to that element.
        x[i, j] = 0  # assign to activate!

@ti.kernel
def paint(t: ti.f32):
    # struct-for syntax: loop over active pixels, inactive pixels are excluded
    for i, j in x:
        x[i, j] = ti.sin(t)

@ti.kernel
def paint2(t: ti.f32):
    # range-for syntax: loop over all pixels, no matter active or not
    for i, j in ti.ndrange(n, n):
        x[i, j] = ti.sin(t)


activate()

gui = ti.GUI('bitmasked', n)
for frame in range(10000):
    paint(frame * 0.05)
    #paint2(frame * 0.05)  # try this!!
    gui.set_image(x)
    gui.show()
