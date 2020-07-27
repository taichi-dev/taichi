import taichi as ti

# declare a 512x512 field whose elements are 3D vectors (RGB channels)
rgb_image = ti.Vector.field(3, dtype=ti.f32, shape=(512, 512))


@ti.kernel  # functions decorated by @ti.kernel will be compiled by Taichi
def render():
    for i, j in rgb_image:  # iterate over each pixels in the 512x512 field
        r = i / 512
        g = j / 512
        b = 0.0
        # set the vector value at [i, j] in the field:
        rgb_image[i, j] = ti.Vector([r, g, b])


gui = ti.GUI('UV', (512, 512))  # create a 512x512 window
while gui.running:
    render()
    gui.set_image(rgb_image)  # display the rendered image
    gui.show()
