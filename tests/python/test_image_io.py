import os

import numpy as np
import pytest

import taichi as ti
from taichi import make_temp_file


# jpg is also supported but hard to test here since it's lossy:
@pytest.mark.parametrize('comp,ext', [(3, 'bmp'), (1, 'png'), (3, 'png'),
                                      (4, 'png')])
@pytest.mark.parametrize('resx,resy', [(201, 173)])
@pytest.mark.parametrize('is_field', [False, True])
@pytest.mark.parametrize('dt', [ti.u8])
@ti.host_arch_only
def test_image_io(resx, resy, comp, ext, is_field, dt):
    if comp != 1:
        shape = (resx, resy, comp)
    else:
        shape = (resx, resy)
    if is_field:
        pixel_t = ti.field(dt, shape)
    pixel = np.random.randint(256, size=shape, dtype=ti.to_numpy_type(dt))
    if is_field:
        pixel_t.from_numpy(pixel)
    fn = make_temp_file(suffix='.' + ext)
    if is_field:
        ti.imwrite(pixel_t, fn)
    else:
        ti.imwrite(pixel, fn)
    pixel_r = ti.imread(fn)
    if comp == 1:
        # from (resx, resy, 1) to (resx, resy)
        pixel_r = pixel_r.reshape((resx, resy))
    assert (pixel_r == pixel).all()
    os.remove(fn)


@pytest.mark.parametrize('comp,ext', [(3, 'png'), (4, 'png')])
@pytest.mark.parametrize('resx,resy', [(91, 81)])
@pytest.mark.parametrize('dt', [ti.f32, ti.f64])
@ti.host_arch_only
def test_image_io_vector(resx, resy, comp, ext, dt):
    shape = (resx, resy)
    pixel = np.random.rand(*shape, comp).astype(ti.to_numpy_type(dt))
    pixel_t = ti.Vector.field(comp, dt, shape)
    pixel_t.from_numpy(pixel)
    fn = make_temp_file(suffix='.' + ext)
    ti.imwrite(pixel_t, fn)
    pixel_r = (ti.imread(fn).astype(ti.to_numpy_type(dt)) + 0.5) / 256.0
    assert np.allclose(pixel_r, pixel, atol=2e-2)
    os.remove(fn)


@pytest.mark.parametrize('comp,ext', [(3, 'png')])
@pytest.mark.parametrize('resx,resy', [(91, 81)])
@pytest.mark.parametrize('dt', [ti.u16, ti.u32, ti.u64])
@ti.host_arch_only
def test_image_io_uint(resx, resy, comp, ext, dt):
    shape = (resx, resy)
    np_type = ti.to_numpy_type(dt)
    # When saving to disk, pixel data will be truncated into 8 bits.
    # Be careful here if you want lossless saving.
    np_max = np.iinfo(np_type).max // 256
    pixel = np.random.randint(256, size=(*shape, comp), dtype=np_type) * np_max
    pixel_t = ti.Vector.field(comp, dt, shape)
    pixel_t.from_numpy(pixel)
    fn = make_temp_file(suffix='.' + ext)
    ti.imwrite(pixel_t, fn)
    pixel_r = ti.imread(fn).astype(np_type) * np_max
    assert (pixel_r == pixel).all()
    os.remove(fn)


@pytest.mark.parametrize('comp', [1, 3])
@pytest.mark.parametrize('resx,resy', [(91, 81)])
@pytest.mark.parametrize('scale', [1, 2, 3])
@ti.host_arch_only
def test_image_resize_sum(resx, resy, comp, scale):
    shape = (resx, resy)
    if comp != 1:
        shape = shape + (comp, )
    old_img = np.random.rand(*shape).astype(np.float32)
    if resx == resy:
        new_img = ti.imresize(old_img, resx * scale)
    else:
        new_img = ti.imresize(old_img, resx * scale, resy * scale)
    assert np.sum(old_img) * scale**2 == ti.approx(np.sum(new_img))
