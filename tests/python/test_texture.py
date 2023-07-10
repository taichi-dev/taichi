from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image
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
def make_texture_2d_r32f(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0), n: ti.i32):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


@ti.kernel
def make_texture_2d_rgba8(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0), n: ti.i32):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


@ti.kernel
def make_texture_3d(tex: ti.types.rw_texture(num_dimensions=3, fmt=ti.Format.r32f, lod=0), n: ti.i32):
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
            warp_uv = uv + ti.Vector([ti.cos(t + uv.x * 5.0), ti.sin(t + uv.y * 5.0)]) * 0.1
            c = ti.math.vec4(0.0)
            if uv.x > 0.5:
                c = tex.sample_lod(warp_uv, 0.0)
            else:
                c = tex.fetch(ti.cast(warp_uv * n, ti.i32), 0)
            pixels[i, j] = [c.r, c.r, c.r]

    n1 = 128
    texture1 = ti.Texture(ti.Format.r32f, (n1, n1))
    n2 = 256
    texture2 = ti.Texture(ti.Format.r32f, (n2, n2))
    texture3 = ti.Texture(ti.Format.rgba8, (n1, n1))

    make_texture_2d_r32f(texture1, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    make_texture_2d_r32f(texture2, n2)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    make_texture_2d_rgba8(texture3, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 2

    paint(0.1, texture1, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 3

    paint(0.2, texture2, n2)
    assert impl.get_runtime().get_num_compiled_functions() == 3

    # (penguinliong) Remember that non-RW textures don't enforce a format so
    # it's the same as the first call to `paint`.
    paint(0.3, texture3, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 3


@test_utils.test(arch=supported_archs_texture_excluding_load_store)
def test_texture_from_field():
    res = (128, 128)
    f = ti.Vector.field(2, ti.f32, res)
    tex = ti.Texture(ti.Format.r32f, res)

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
    tex = ti.Texture(ti.Format.r32f, res)

    @ti.kernel
    def init_taichi_logo_ndarray(f: ti.types.ndarray(ndim=2)):
        for i, j in f:
            f[i, j] = [taichi_logo(ti.Vector([i / res[0], j / res[1]])), 0]

    init_taichi_logo_ndarray(f)
    tex.from_ndarray(f)


@test_utils.test(arch=supported_archs_texture)
def test_texture_3d():
    res = (32, 32, 32)
    tex = ti.Texture(ti.Format.r32f, res)

    make_texture_3d(tex, res[0])


@test_utils.test(arch=supported_archs_texture)
def test_from_to_image():
    url = "https://github.com/taichi-dev/taichi/blob/master/misc/logo.png?raw=true"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    tex = ti.Texture(ti.Format.rgba8, img.size)

    tex.from_image(img)
    out = tex.to_image()

    assert (np.asarray(out) == np.asarray(img.convert("RGB"))).all()


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_2d_struct_for():
    res = (128, 128)
    tex = ti.Texture(ti.Format.r32f, res)
    arr = ti.ndarray(ti.f32, res)

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    @ti.kernel
    def read(tex: ti.types.texture(num_dimensions=2), arr: ti.types.ndarray()):
        for i, j in arr:
            arr[i, j] = tex.fetch(ti.Vector([i, j]), 0).x

    write(tex)
    read(tex, arr)
    assert arr.to_numpy().sum() == 128 * 128


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_2d_struct_for_dim_check():
    tex = ti.Texture(ti.Format.r32f, (32, 32, 32))

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="RWTextureType dimension mismatch for argument tex: expected 2, got 3",
    ) as e:
        write(tex)


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_wrong_fmt():
    tex = ti.Texture(ti.Format.rgba8, (32, 32))

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="RWTextureType format mismatch for argument tex: expected Format.r32f, got Format.rgba8",
    ) as e:
        write(tex)


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_wrong_ndim():
    tex = ti.Texture(ti.Format.rgba8, (32, 32))

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=1, fmt=ti.Format.rgba8, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="RWTextureType dimension mismatch for argument tex: expected 1, got 2",
    ) as e:
        write(tex)
