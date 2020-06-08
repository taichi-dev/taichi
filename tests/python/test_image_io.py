import taichi as ti
import numpy as np
import pytest
import os


# jpg is also supported but hard to test here since it's lossy:
@pytest.mark.parametrize('resx,resy', [(201, 173)])
@pytest.mark.parametrize('comp,ext', [(3, 'bmp'), (1, 'png'), (3, 'png'), (4, 'png')])
@ti.host_arch_only
def test_image_io(resx, resy, comp, ext):
    from tempfile import mkstemp
    if comp != 1:
        pixel = np.random.randint(256, size=(resx, resy, comp), dtype=np.uint8)
    else:
        pixel = np.random.randint(256, size=(resx, resy), dtype=np.uint8)
    fn = mkstemp(suffix='.' + ext)[1]
    ti.imwrite(pixel, fn)
    pixel_r = ti.imread(fn)
    if comp == 1:
        # from (resx, resy, 1) to (resx, resy)
        pixel_r = pixel_r.reshape((resx, resy))
    print(pixel_r)
    print('=====')
    print(pixel)
    assert (pixel_r == pixel).all()
    os.remove(fn)
