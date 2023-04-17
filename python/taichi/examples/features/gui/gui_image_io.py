import os

import taichi as ti

ti.init(arch=ti.cpu)

pixel = ti.field(ti.u8, shape=(512, 512, 3))


@ti.kernel
def paint():
    for I in ti.grouped(pixel):
        pixel[I] = ti.u8(ti.random() * 255)


paint()
pixel = pixel.to_numpy()
ti.tools.imshow(pixel, "Random Generated")
for ext in ["bmp", "png", "jpg"]:
    fn = "taichi-example-random-img." + ext
    ti.tools.imwrite(pixel, fn)
    pixel_r = ti.tools.imread(fn)
    if ext != "jpg":
        assert (pixel_r == pixel).all()
    else:
        ti.tools.imshow(pixel_r, "JPEG Read Result")
    os.remove(fn)
