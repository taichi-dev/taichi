import taichi as ti  # make sure you have done "pip3 install taichi"

# declare a 512x512x3 field whose elements are 32-bit float-point numbers
rgb_image = ti.field(dtype=float, shape=(512, 512, 3))


@ti.kernel  # functions decorated by @ti.kernel will be compiled by Taichi
def render():
    # iterate through 512x512 pixels in parallel
    for i, j in ti.ndrange(512, 512):
        r = i / 512
        g = j / 512
        rgb_image[i, j, 0] = r  # red channel, from 0.0 to 1.0
        rgb_image[i, j, 1] = g  # green channel, from 0.0 to 1.0


gui = ti.GUI('UV', (512, 512))  # create a 512x512 window
while gui.running:
    render()
    gui.set_image(rgb_image)  # display the field as an image
    gui.show()
