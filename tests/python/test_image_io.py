import taichi as ti
import numpy as np
import pytest
import os


# jpg is also supported but hard to test here since it's lossy:
@pytest.mark.parametrize('comp,ext', [(3, 'bmp'), (1, 'png'), (3, 'png'),
                                      (4, 'png')])
@pytest.mark.parametrize('resx,resy', [(201, 173)])
@pytest.mark.parametrize('is_tensor', [False, True])
@pytest.mark.parametrize('dt', [ti.u8])
@ti.host_arch_only
def test_image_io(resx, resy, comp, ext, is_tensor, dt):
    from tempfile import mkstemp
    if comp != 1:
        shape = (resx, resy, comp)
    else:
        shape = (resx, resy)
    if is_tensor:
        pixel_t = ti.var(dt, shape)
    pixel = np.random.randint(256, size=shape, dtype=ti.to_numpy_type(dt))
    if is_tensor:
        pixel_t.from_numpy(pixel)
    fn = mkstemp(suffix='.' + ext)[1]
    if is_tensor:
        ti.imwrite(pixel_t, fn)
    else:
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
