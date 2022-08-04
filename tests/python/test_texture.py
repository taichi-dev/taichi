from io import BytesIO

import numpy as np
import requests
from PIL import Image, ImageChops
from taichi.lang import impl

import taichi as ti
from tests import test_utils

supported_archs_texture = [ti.vulkan]
supported_archs_texture_excluding_load_store = [ti.vulkan, ti.opengl]


@ti.func
def taichi_logo(pos: ti.template(), scale: float = 1 / 1.11):
    p = (pos - 0.5) / scale + 0.5
    ret = -1
    if not (p - 0.50).norm_sqr() <= 0.52**2:
        if ret == -1:
            ret = 0
    if not (p - 0.50).norm_sqr() <= 0.495**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 0
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 0
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 1
    if p[0] < 0.5:
        if ret == -1:
            ret = 1
    else:
        if ret == -1:
            ret = 0
    return 1 - ret


@ti.kernel
def make_texture_2d(tex: ti.types.rw_texture(
    num_dimensions=2, num_channels=1, channel_format=ti.f32, lod=0), n: ti.i32
                    ):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


@ti.kernel
def make_texture_3d(tex: ti.types.rw_texture(
    num_dimensions=3, num_channels=1, channel_format=ti.f32, lod=0), n: ti.i32
                    ):
    for i, j, k in ti.ndrange(n, n, n):
        div = ti.cast(i / n, ti.f32)
        if div > 0.5:
            tex.store(ti.Vector([i, j, k]), ti.Vector([1.0, 0.0, 0.0, 0.0]))
        else:
            tex.store(ti.Vector([i, j, k]), ti.Vector([0.5, 0.0, 0.0, 0.0]))


@test_utils.test(arch=supported_archs_texture)
def test_texture_compiled_functions():
    res = (512, 512)
    pixels = ti.Vector.field(3, dtype=float, shape=res)

    @ti.kernel
    def paint(t: ti.f32, tex: ti.types.texture(num_dimensions=2), n: ti.i32):
        for i, j in pixels:
            uv = ti.Vector([i / res[0], j / res[1]])
            warp_uv = uv + ti.Vector(
                [ti.cos(t + uv.x * 5.0),
                 ti.sin(t + uv.y * 5.0)]) * 0.1
            c = ti.math.vec4(0.0)
            if uv.x > 0.5:
                c = tex.sample_lod(warp_uv, 0.0)
            else:
                c = tex.fetch(ti.cast(warp_uv * n, ti.i32), 0)
            pixels[i, j] = [c.r, c.r, c.r]

    n1 = 128
    texture1 = ti.Texture(ti.f32, 1, (n1, n1))
    n2 = 256
    texture2 = ti.Texture(ti.f32, 1, (n2, n2))

    make_texture_2d(texture1, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    make_texture_2d(texture2, n2)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    paint(0.1, texture1, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 2

    paint(0.2, texture2, n2)
    assert impl.get_runtime().get_num_compiled_functions() == 2


@test_utils.test(arch=supported_archs_texture_excluding_load_store)
def test_texture_from_field():
    res = (128, 128)
    f = ti.Vector.field(2, ti.f32, res)
    tex = ti.Texture(ti.f32, 1, res)

    @ti.kernel
    def init_taichi_logo_field():
        for i, j in f:
            f[i, j] = [taichi_logo(ti.Vector([i / res[0], j / res[1]])), 0]

    init_taichi_logo_field()
    tex.from_field(f)


@test_utils.test(arch=supported_archs_texture_excluding_load_store)
def test_texture_from_ndarray():
    res = (128, 128)
    f = ti.Vector.ndarray(2, ti.f32, res)
    tex = ti.Texture(ti.f32, 1, res)

    @ti.kernel
    def init_taichi_logo_ndarray(f: ti.types.ndarray(field_dim=2)):
        for i, j in f:
            f[i, j] = [taichi_logo(ti.Vector([i / res[0], j / res[1]])), 0]

    init_taichi_logo_ndarray(f)
    tex.from_ndarray(f)


@test_utils.test(arch=supported_archs_texture)
def test_texture_3d():
    res = (32, 32, 32)
    tex = ti.Texture(ti.f32, 1, res)

    make_texture_3d(tex, res[0])


@test_utils.test(arch=supported_archs_texture)
def test_from_to_image():
    url = 'https://github.com/taichi-dev/taichi/blob/master/misc/logo.png?raw=true'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    tex = ti.Texture(ti.u8, 4, img.size)

    tex.from_image(img)
    out = tex.to_image()

    assert (np.asarray(out) == np.asarray(img.convert('RGB'))).all()
