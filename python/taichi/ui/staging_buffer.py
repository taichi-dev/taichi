from collections import defaultdict
import numpy as np
from taichi.lang.impl import ndarray
from taichi.lang.kernel_impl import kernel
from taichi.lang.matrix import Vector
from taichi.types import ndarray_type
from taichi.types.annotations import template
from taichi.types.primitive_types import f32, u8, u32

import taichi as ti
from taichi.ui.utils import GGUIException, get_field_info

depth_ndarray_cache = {}

def get_depth_ndarray(window):
    if window not in depth_ndarray_cache:
        w, h = window.get_window_shape()
        depth_arr = ndarray(dtype=ti.f32, shape=w * h)
        depth_ndarray_cache[window] = depth_arr
    return depth_ndarray_cache[window]


def validate_input_field(f, name, min_comp, max_comp):
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
    if not min_comp <= f.n <= max_comp:
        if min_comp == max_comp:
            raise Exception(f"{name} can only have {min_comp} component")
        else:
            raise Exception(f"{name} can only have {min_comp} to {max_comp} vector components")

def validate_input_arr(f, name, min_comp, max_comp):
    if f.dtype != np.float32:
        raise Exception(f"numpy array {name} needs to have dtype np.float32")
    if len(f.shape) != 2:
        raise Exception(f"the shape of numpy array {name} needs to be 2-dimensional")
    if not min_comp <= f.shape[-1] <= max_comp:
        if min_comp == max_comp:
            raise Exception(f"{name} can only have {min_comp} component")
        else:
            raise Exception(f"numpy array {name} can only have {min_comp} to {max_comp} vector components")


def validate_input(f, name, min_comp, max_comp):
    if isinstance(f, np.ndarray):
        validate_input_arr(f, name, min_comp, max_comp)
    else:
        validate_input_field(f, name, min_comp, max_comp)

@kernel
def copy_field_to_vbo(vbo: template(), src: template(), offset: template(),
                num_components: template()):
    for i in src:
        for c in ti.static(range(num_components)):
            vbo[i][offset + c] = src[i][c]

@kernel
def copy_arr_to_vbo(vbo: template(), src: ndarray_type.ndarray(), offset: template(),
                num_components: template()):
    for i in vbo:
        for c in ti.static(range(num_components)):
            vbo[i][offset + c] = src[i, c]

def copy_to_vbo(vbo, src, offset):
    if isinstance(src, np.ndarray):
        copy_arr_to_vbo(vbo, src, offset, src.shape[-1])
    else:
        copy_field_to_vbo(vbo, src, offset, src.n)

@kernel
def fill_vbo(vbo: template(), value: f32, offset: template(),
             num_components: template()):
    for i in vbo:
        for c in ti.static(range(num_components)):
            vbo[i][offset + c] = value

class Vbo:
    def __init__(self, N):
        pos = 3
        normal = 3
        tex_coord = 2
        color = 4
        vertex_stride = pos + normal + tex_coord + color

        self.vbo = Vector.field(vertex_stride, f32, shape=(N, ))
        self.N = N
        
    def set_positions(self, positions):
        validate_input(positions, "vertex positions", 2, 3)
        copy_to_vbo(self.vbo, positions, 0)

    def set_normals(self, normals):
        validate_input(normals, "normals", 3, 3)
        copy_to_vbo(self.vbo, normals, 3)

    def set_texcoords(self, texcoords):
        validate_input(texcoords, "texcoords", 2, 2)
        copy_to_vbo(self.vbo, texcoords, 6)

    def set_colors(self, colors):
        validate_input(colors, "colors", 3, 4)
        copy_to_vbo(self.vbo, colors, 8)
        if isinstance(colors, np.ndarray):
            if colors.n == 3:
                fill_vbo(self.vbo, 1.0, 11, 1)
        else:
            if colors.shape[-1] == 3:
                fill_vbo(self.vbo, 1.0, 11, 1)

    def get_field_info(self):
        return get_field_info(self.vbo)

class VboPool:
    def __init__(self, max_pool_size):
        """Create a VBO pool for a window."""
        self.count = 0
        self.pool = defaultdict(list) # number of vertices -> vbo
        self.max_size = max_pool_size
        self.allocated = []

    def allocate(self, N):
        if self.count >= self.max_size:
            msg = """The VBO pool refuse to allocate another VBO because the
            pool has been saturated. To solve this problem, please try:
            1. Ensure you called `ti.ui.Window.show()` at the end of the frame;
            2. Or less likely, set a larger pool size via
            `ti.ui.set_max_vbo_pool_size`, or remove this allocation constraint
            by setting it to 0.
            """
            raise GGUIException(msg)
        vbo = Vbo(N)
        self.count += 1
        return vbo

    def acquire(self, N):
        """Acquire a VBO to contain N vertices."""
        pool = self.pool[N]
        vbo = None
        if pool:
            vbo = pool.pop()
        else:
            vbo = self.allocate(N)
        self.allocated += [vbo]
        return vbo

    def reset(self):
        """Release VBO occupations after the rendering of this frame. Should be
        called only by `ti.ui.Window.show()`.
        """
        for vbo in self.allocated:
            self.pool[vbo.N] += [vbo]
        self.allocated.clear()

DEFAULT_VBO_POOL_SIZE = 100
vbo_pool = VboPool(DEFAULT_VBO_POOL_SIZE)
def set_max_vbo_pool_size(pool_size):
    global vbo_pool
    vbo_pool = VboPool(pool_size)

def get_vbo(positions):
    N = positions.shape[0]
    return vbo_pool.acquire(N)


def reset_vbo_pool():
    vbo_pool.reset()

@ti.kernel
def copy_image_f32_to_rgba8(src: ti.template(), dst: ti.template(),
                            num_components: ti.template()):
    for i, j in src:
        px = ti.Vector([0, 0, 0, 0xff], dt=u32)
        for k in ti.static(range(num_components)):
            c = src[i, j][k]
            c = max(0.0, min(1.0, c))
            c = c * 255
            px[k] = ti.cast(c, u32)
        pack = (px[0] << 0 | px[1] << 8 | px[2] << 16 | px[3] << 24)
        dst[i, j] = pack


@ti.kernel
def copy_image_u8_to_rgba8(src: ti.template(), dst: ti.template(),
                           num_components: ti.template()):
    for i, j in src:
        px = ti.Vector([0, 0, 0, 0xff], dt=u32)
        for k in ti.static(range(num_components)):
            px[k] = ti.cast(src[i, j][k], u32)
        pack = (px[0] << 0 | px[1] << 8 | px[2] << 16 | px[3] << 24)
        dst[i, j] = pack


# ggui renderer always assumes the input image to be u8 RGBA
# if the user input is not in this format, a staging ti field is needed
image_field_cache = {}


def to_rgba8(image):
    if not hasattr(image, 'n') or image.m != 1:
        raise Exception(
            'the input image needs to be a Vector field (matrix with 1 column)'
        )
    if len(image.shape) != 2:
        raise Exception(
            "the shape of the image must be of the form (width,height)")

    if image not in image_field_cache:
        staging_img = ti.field(u32, image.shape)
        image_field_cache[image] = staging_img
    else:
        staging_img = image_field_cache[image]

    if image.dtype == u8:
        copy_image_u8_to_rgba8(image, staging_img, image.n)
    elif image.dtype == f32:
        copy_image_f32_to_rgba8(image, staging_img, image.n)
    else:
        raise Exception("dtype of input image must either be u8 or f32")
    return staging_img
