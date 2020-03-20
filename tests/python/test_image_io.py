import taichi as ti
import numpy as np
import os


def test_image_io():
    pixel = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    for ext in [
            'bmp', 'png'
    ]:  # jpg is also supported but hard to test here since it's lossy
        fn = 'taichi-image-io-test.' + ext
        ti.imwrite(pixel, fn)
        pixel_r = ti.imread(fn)
        assert (pixel_r == pixel).all()
        os.remove(fn)
