import os

import numpy as np

import taichi as ti
from tests import test_utils
from taichi.math import *


@test_utils.test(arch=[ti.cuda])
def test_ad_shading():
    f16vec3 = ti.types.vector(3, ti.f16)
    f16vec2 = ti.types.vector(2, ti.f16)

    @ti.func
    def logistics(x, scale, shift):
        return 1.0 / (1.0 + ti.exp(-scale * (x - shift)))

    @ti.func
    def gl_transform_uv(value: ti.f32, mode: int) -> ti.f32:
        tvalue = value
        if mode == 0:  # CLAMP_TO_EDGE
            tvalue = clamp(value, 0.0, 1.0)
        elif mode == 1:  # CLAMP_TO_BORDER
            tvalue = -1.0
        elif mode == 2:  # REPEAT
            tvalue = fract(value)
        elif mode == 3:  # MIRRORED_REPEAT
            tvalue = 1.0 - abs(fract(0.5 * value) - 0.5) * 2.0
        return tvalue

    @ti.func
    def color_tf(encoded):
        rgb = logistics(encoded, 1.0, 0.0) * 1.3 - 0.15

        return rgb

    @ti.func
    def process_uv(uv: vec2, mode: ivec2) -> vec2:
        return vec2(gl_transform_uv(uv.x, mode.x), gl_transform_uv(uv.y, mode.y))

    @ti.func
    def fetch_texel(tex: ti.template(), tex_id: ti.i32, iuv: ivec2, dir: ti.i32) -> vec3:
        iuv_clamped = clamp(iuv, 0, tex.shape[2] - 1)
        texel = tex[tex_id, dir, iuv_clamped.y, iuv_clamped.x]
        color = color_tf(texel)

        return color

    @ti.kernel
    def ad_shade(
        tex: ti.types.ndarray(vec3, ndim=4, needs_grad=True),
        uv_mode: ti.types.ndarray(ivec2, ndim=1),
        normal_adjustment: ti.types.ndarray(ti.f32, ndim=3),
        ad_beauty: ti.types.ndarray(f16vec3, ndim=3, needs_grad=True),
        stencil_buffer: ti.types.ndarray(ti.i16, ndim=3),
        uv_buffer: ti.types.ndarray(f16vec2, ndim=4),
        offset_buffer: ti.types.ndarray(vec2, ndim=2),
    ):
        # Shade
        for view, x, y, i in ti.ndrange(4, 1024, 1024, 64):
            # FIXME: remove this cast once #7946 is fixed
            stencil = ti.cast(stencil_buffer[view, y, x], ti.i32)
            if stencil < 0x7FFF:
                uv00 = ti.cast(uv_buffer[view, y, x, 0], ti.f32)
                uv01 = ti.cast(uv_buffer[view, y, x, 1], ti.f32)
                uv10 = ti.cast(uv_buffer[view, y, x, 2], ti.f32)
                dUVdx = uv10 - uv00
                dUVdy = uv01 - uv00
                offset = offset_buffer[view, i]
                footprint = ti.min(1.5, 1.0 / normal_adjustment[view, y, x])
                uv = uv00 + footprint * dUVdx * (offset.x - 0.5) + footprint * dUVdy * (offset.y - 0.5)
                uv = process_uv(uv, uv_mode[stencil])
                tex_size = tex.shape[2]
                uv = uv * (tex_size - 1)
                iuv = ti.floor(uv)
                fuv = uv - iuv
                color = vec3(0.0)
                if all(uv >= 0.0):
                    color00 = fetch_texel(tex, stencil, ti.cast(uv, ti.i32), view)
                    color01 = fetch_texel(tex, stencil, ti.cast(uv, ti.i32) + ivec2(0, 1), view)
                    color10 = fetch_texel(tex, stencil, ti.cast(uv, ti.i32) + ivec2(1, 0), view)
                    color11 = fetch_texel(tex, stencil, ti.cast(uv, ti.i32) + ivec2(1, 1), view)
                    color = mix(
                        mix(color00, color01, fuv.y),
                        mix(color10, color11, fuv.y),
                        fuv.x,
                    )
                ad_beauty[view, y, x] += color / 64

    tex = ti.ndarray(vec3, shape=[4, 128, 128, 3], needs_grad=True)
    uv_mode = ti.ndarray(ivec2, shape=[2])
    normal_adjustment = ti.ndarray(ti.f32, shape=[4, 1024, 1024])
    ad_beauty = ti.ndarray(f16vec3, shape=[1024, 1024, 3], needs_grad=True)
    stencil_buffer = ti.ndarray(ti.i16, shape=[4, 1024, 1024])
    uv_buffer = ti.ndarray(f16vec2, shape=[1024, 1024, 3, 2])
    offset_buffer = ti.ndarray(vec2, shape=[64, 2])

    ad_shade.grad(tex, uv_mode, normal_adjustment, ad_beauty, stencil_buffer, uv_buffer, offset_buffer)
