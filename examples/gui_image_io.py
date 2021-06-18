import taichi as ti
import os

pixel = ti.field(ti.u8, shape=(512, 512, 3))


@ti.kernel
def paint():
    for I in ti.grouped(pixel):
        pixel[I] = ti.random() * 255


paint()
pixel = pixel.to_numpy()
ti.imshow(pixel, 'Random Generated')
for ext in ['bmp', 'png', 'jpg']:
    fn = 'taichi-example-random-img.' + ext
    ti.imwrite(pixel, fn)
    pixel_r = ti.imread(fn)
    if ext != 'jpg':
        assert (pixel_r == pixel).all()
    else:
        ti.imshow(pixel_r, 'JPEG Read Result')
    os.remove(fn)
