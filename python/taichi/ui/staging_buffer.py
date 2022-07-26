from taichi.lang.impl import ndarray
from taichi.lang.kernel_impl import kernel
from taichi.lang.matrix import Vector
from taichi.types.annotations import template
from taichi.types.primitive_types import f32, u8

import taichi as ti

vbo_field_cache = {}
depth_ndarray_cache = {}


def get_vbo_field(vertices):
    if vertices not in vbo_field_cache:
        N = vertices.shape[0]
        pos = 3
        normal = 3
        tex_coord = 2
        color = 4
        vertex_stride = pos + normal + tex_coord + color
        vbo = Vector.field(vertex_stride, f32, shape=(N, ))
        vbo_field_cache[vertices] = vbo
        return vbo
    return vbo_field_cache[vertices]


def get_depth_ndarray(window):
    if window not in depth_ndarray_cache:
        w, h = window.get_window_shape()
        depth_arr = ndarray(dtype=ti.f32, shape=w * h)
        depth_ndarray_cache[window] = depth_arr
    return depth_ndarray_cache[window]


@kernel
def copy_to_vbo(vbo: template(), src: template(), offset: template(),
                num_components: template()):
    for i in src:
        for c in ti.static(range(num_components)):
            vbo[i][offset + c] = src[i][c]


@kernel
def fill_vbo(vbo: template(), value: template(), offset: template(),
             num_components: template()):
    for i in vbo:
        for c in ti.static(range(num_components)):
            vbo[i][offset + c] = value


def validate_input_field(f, name):
    if f.dtype != f32:
        raise Exception(f"{name} needs to have dtype f32")
    if hasattr(f, 'n'):
        if f.m != 1:
            raise Exception(
                f'{name} needs to be a Vector field (matrix with 1 column)')
    else:
        raise Exception(f'{name} needs to be a Vector field')
    if len(f.shape) != 1:
        raise Exception(f"the shape of {name} needs to be 1-dimensional")


def copy_vertices_to_vbo(vbo, vertices):
    validate_input_field(vertices, "vertices")
    if not 2 <= vertices.n <= 3:
        raise Exception('vertices can only be 2D or 3D vector fields')
    copy_to_vbo(vbo, vertices, 0, vertices.n)


def copy_normals_to_vbo(vbo, normals):
    validate_input_field(normals, "normals")
    if normals.n != 3:
        raise Exception('normals can only be 3D vector fields')
    copy_to_vbo(vbo, normals, 3, normals.n)


def copy_texcoords_to_vbo(vbo, texcoords):
    validate_input_field(texcoords, "texcoords")
    if texcoords.n != 2:
        raise Exception('texcoords can only be 3D vector fields')
    copy_to_vbo(vbo, texcoords, 6, texcoords.n)


def copy_colors_to_vbo(vbo, colors):
    validate_input_field(colors, "colors")
    if colors.n != 3 and colors.n != 4:
        raise Exception('colors can only be 3D/4D vector fields')
    copy_to_vbo(vbo, colors, 8, colors.n)
    if colors.n == 3:
        fill_vbo(vbo, ti.cast(1.0, ti.f32), 11, 1)


@ti.kernel
def copy_image_f32_to_u8(src: ti.template(), dst: ti.template(),
                         num_components: ti.template()):
    for i, j in src:
        for k in ti.static(range(num_components)):
            c = src[i, j][k]
            c = max(0.0, min(1.0, c))
            c = c * 255
            dst[i, j][k] = ti.cast(c, u8)
        if num_components < 4:
            # alpha channel
            dst[i, j][3] = u8(255)


@ti.kernel
def copy_image_u8_to_u8(src: ti.template(), dst: ti.template(),
                        num_components: ti.template()):
    for i, j in src:
        for k in ti.static(range(num_components)):
            dst[i, j][k] = ti.cast(src[i, j][k], ti.u8)
        if num_components < 4:
            # alpha channel
            dst[i, j][3] = u8(255)


# ggui renderer always assumes the input image to be u8 RGBA
# if the user input is not in this format, a staging ti field is needed
image_field_cache = {}


def to_u8_rgba(image):
    if not hasattr(image, 'n') or image.m != 1:
        raise Exception(
            'the input image needs to be a Vector field (matrix with 1 column)'
        )
    if len(image.shape) != 2:
        raise Exception(
            "the shape of the image must be of the form (width,height)")

    if image.dtype == u8 and image.n == 4:
        # already in the desired format
        return image

    if image not in image_field_cache:
        staging_img = Vector.field(4, u8, image.shape)
        image_field_cache[image] = staging_img
    else:
        staging_img = image_field_cache[image]

    if image.dtype == u8:
        copy_image_u8_to_u8(image, staging_img, image.n)
    elif image.dtype == f32:
        copy_image_f32_to_u8(image, staging_img, image.n)
    else:
        raise Exception("dtype of input image must either be u8 or f32")
    return staging_img
