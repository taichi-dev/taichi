import numpy as np
from taichi.types import ndarray as ndarray_type
from taichi.lang import ops
from taichi.lang._texture import Texture
from taichi.lang.impl import ndarray
from taichi.lang.kernel_impl import kernel
from taichi.types.annotations import template
from taichi.types.primitive_types import f32, u8, u32

import taichi as ti

vbo_field_cache = {}
depth_ndarray_cache = {}
indices_ndarray_cache = {}
transforms_ndarray_cache = {}


def get_depth_ndarray(window):
    if window not in depth_ndarray_cache:
        w, h = window.get_window_shape()
        depth_arr = ndarray(dtype=ti.f32, shape=w * h)
        depth_ndarray_cache[window] = depth_arr
    return depth_ndarray_cache[window]


def get_vbo_field(vertices):
    if vertices not in vbo_field_cache:
        N = vertices.shape[0]
        pos = 3
        normal = 3
        tex_coord = 2
        color = 4
        vertex_stride = pos + normal + tex_coord + color
        vbo = np.ndarray((N, vertex_stride), dtype=np.float32)
        vbo_field_cache[vertices] = vbo
        return vbo
    return vbo_field_cache[vertices]


def get_vbo_field_v2(vertices):
    N = vertices.shape[0]
    pos = 3
    normal = 3
    tex_coord = 2
    color = 4
    vertex_stride = pos + normal + tex_coord + color
    vbo = np.ndarray((N, vertex_stride), dtype=np.float32)
    return vbo


def get_indices_field(indices):
    if isinstance(indices, np.ndarray):
        return indices
    indices_arr = indices.to_numpy()
    indices_ndarray_cache[indices] = indices_arr
    return indices_arr


def get_indices_field_v2(indices):
    if isinstance(indices, np.ndarray):
        return indices
    indices_arr = indices.to_numpy()
    return indices_arr


def get_transforms_field(transforms):
    if isinstance(transforms, np.ndarray):
        return transforms
    transforms_arr = transforms.to_numpy()
    transforms_ndarray_cache[transforms] = transforms_arr
    return transforms_arr


def get_transforms_field_v2(transforms):
    if isinstance(transforms, np.ndarray):
        return transforms
    transforms_arr = transforms.to_numpy()
    return transforms_arr


@kernel
def copy_all_to_vbo(
    vbo: ndarray_type(element_dim=1),
    vertex: template(),
    normal: template(),
    texcoords: template(),
    color: template(),
):
    for i in vertex:
        if ti.static(vertex.n == 3):
            vbo[i][0:3] = vertex[i]
        else:
            vbo[i][0:2] = vertex[i]
            vbo[i][3] = 0.0
        if ti.static(normal != 0):
            vbo[i][3:6] = normal[i]
        if ti.static(texcoords != 0):
            vbo[i][6:8] = texcoords[i]
        if ti.static(color != 0):
            if ti.static(color.n == 3):
                vbo[i][8:11] = color[i]
                vbo[i][11] = 1.0
            else:
                vbo[i][8:12] = color[i]


@kernel
def copy_all_to_vbo_particle(  # The vbo copy function used for vertices of particles and circles
    vbo: ndarray_type(element_dim=1),
    vertex: template(),
    radius: template(),
    color: template(),
):
    for i in vertex:
        if ti.static(vertex.n == 3):
            vbo[i][0:3] = vertex[i]
        else:
            vbo[i][0:2] = vertex[i]
            vbo[i][3] = 0.0
        if ti.static(radius != 0):
            vbo[i][3] = radius[i]
        if ti.static(color != 0):
            if ti.static(color.n == 3):
                vbo[i][8:11] = color[i]
                vbo[i][11] = 1.0
            else:
                vbo[i][8:12] = color[i]


@ti.kernel
def copy_texture_to_rgba8(
    src: ti.types.texture(num_dimensions=2),
    dst: ti.types.ndarray(),
    w: ti.i32,
    h: ti.i32,
):
    for i, j in ti.ndrange(w, h):
        c = src.fetch(ti.Vector([i, j]), 0)
        c = ops.max(0.0, ops.min(1.0, c))
        c = c * 255
        px = ti.cast(c, u32)
        dst[i, j] = px[0] << 0 | px[1] << 8 | px[2] << 16 | px[3] << 24


@ti.kernel
def copy_image_f32_to_rgba8(
    src: ti.template(),
    dst: ti.types.ndarray(),
    num_components: ti.template(),
    gray_scale: ti.template(),
):
    for i, j in ti.ndrange(src.shape[0], src.shape[1]):
        px = ti.Vector([0, 0, 0, 0xFF], dt=u32)
        if ti.static(gray_scale):
            c = 0.0
            c = src[i, j]
            c = ops.max(0.0, ops.min(1.0, c))
            c = c * 255
            px[0] = px[1] = px[2] = ti.cast(c, u32)
        else:
            for k in ti.static(range(num_components)):
                c = 0.0
                if ti.static(len(src.shape) == 3):
                    # 3D field source image
                    c = src[i, j, k]
                else:
                    # 2D vector field source image
                    c = src[i, j][k]
                c = ops.max(0.0, ops.min(1.0, c))
                c = c * 255
                px[k] = ti.cast(c, u32)
        pack = px[0] << 0 | px[1] << 8 | px[2] << 16 | px[3] << 24
        dst[i, j] = pack


@ti.kernel
def copy_image_f32_to_rgba8_np(
    src: ti.types.ndarray(),
    dst: ti.types.ndarray(),
    num_components: ti.template(),
    gray_scale: ti.template(),
):
    for I in ti.grouped(src):
        i, j = I[0], I[1]
        px = ti.Vector([0, 0, 0, 0xFF], dt=u32)
        if ti.static(gray_scale):
            c = 0.0
            c = src[i, j]
            c = ops.max(0.0, ops.min(1.0, c))
            c = c * 255
            px[0] = px[1] = px[2] = ti.cast(c, u32)
        else:
            for k in ti.static(range(num_components)):
                c = src[i, j, k]
                c = ops.max(0.0, ops.min(1.0, c))
                c = c * 255
                px[k] = ti.cast(c, u32)
        pack = px[0] << 0 | px[1] << 8 | px[2] << 16 | px[3] << 24
        dst[i, j] = pack


@ti.kernel
def copy_image_u8_to_rgba8(
    src: ti.template(),
    dst: ti.types.ndarray(),
    num_components: ti.template(),
    gray_scale: ti.template(),
):
    for i, j in ti.ndrange(src.shape[0], src.shape[1]):
        px = ti.Vector([0, 0, 0, 0xFF], dt=u32)
        if ti.static(gray_scale):
            px[0] = px[1] = px[2] = ti.cast(src[i, j], u32)
        else:
            for k in ti.static(range(num_components)):
                if ti.static(len(src.shape) == 3):
                    # 3D field source image
                    px[k] = ti.cast(src[i, j, k], u32)
                else:
                    # 2D vector field source image
                    px[k] = ti.cast(src[i, j][k], u32)
        pack = px[0] << 0 | px[1] << 8 | px[2] << 16 | px[3] << 24
        dst[i, j] = pack


@ti.kernel
def copy_image_u8_to_rgba8_np(
    src: ti.types.ndarray(),
    dst: ti.types.ndarray(),
    num_components: ti.template(),
    gray_scale: ti.template(),
):
    for I in ti.grouped(src):
        i, j = I[0], I[1]
        px = ti.Vector([0, 0, 0, 0xFF], dt=u32)
        if ti.static(gray_scale):
            px[0] = px[1] = px[2] = ti.cast(src[i, j], u32)
        else:
            for k in ti.static(range(num_components)):
                px[k] = ti.cast(src[i, j, k], u32)
        pack = px[0] << 0 | px[1] << 8 | px[2] << 16 | px[3] << 24
        dst[i, j] = pack


# ggui renderer always assumes the input image to be u8 RGBA
# if the user input is not in this format, a staging ti field is needed
image_field_cache = {}


def to_rgba8(image):
    is_texture = isinstance(image, Texture)
    is_grayscale = not hasattr(image, "n") and len(image.shape) == 2
    is_numpy = isinstance(image, np.ndarray)
    is_non_grayscale_field = (hasattr(image, "n") and image.m == 1) or len(image.shape) == 3

    if not is_texture and not is_grayscale and not is_numpy and not is_non_grayscale_field:
        raise Exception(
            "the input image needs to be either:\n"
            "a Vector field (matrix with 1 column)\n"
            "a 2D(grayscale)/3D field\n"
            "a 2D(grayscale)/3D numpy ndarray\n"
            "a texture"
        )
    channels = 3

    if not is_grayscale:
        if len(image.shape) == 2:
            channels = image.n
        elif len(image.shape) == 3:
            channels = image.shape[2]
        else:
            raise Exception("the shape of the image must be of the form (width,height) or (width,height,channels)")

    staging_key = image.shape[0:2] if is_numpy else image

    if staging_key not in image_field_cache:
        staging_img = np.ndarray(image.shape[0:2], dtype=np.uint32)
        image_field_cache[staging_key] = staging_img
    else:
        staging_img = image_field_cache[staging_key]

    if is_texture:
        copy_texture_to_rgba8(image, staging_img, *image.shape[0:2])
    elif is_numpy:
        if image.dtype == np.uint8:
            copy_image_u8_to_rgba8_np(image, staging_img, channels, is_grayscale)
        elif image.dtype == np.float32:
            copy_image_f32_to_rgba8_np(image, staging_img, channels, is_grayscale)
        else:
            raise Exception("dtype of input image must either be u8 or f32")
    else:
        if image.dtype == u8:
            copy_image_u8_to_rgba8(image, staging_img, channels, is_grayscale)
        elif image.dtype == f32:
            copy_image_f32_to_rgba8(image, staging_img, channels, is_grayscale)
        else:
            raise Exception("dtype of input image must either be u8 or f32")

    return staging_img
