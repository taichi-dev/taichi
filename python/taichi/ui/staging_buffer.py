from taichi.core import ti_core as _ti_core
from taichi.core.primitive_types import f32
from taichi.lang.impl import default_cfg, field, static
from taichi.lang.ndrange import ndrange
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel
from taichi.lang.matrix import Vector
from taichi.lang.ops import atomic_add, get_addr

from .utils import get_field_info

import taichi as ti


vbo_field_cache = {}

def get_vbo_field(vertices):
    if vertices not in vbo_field_cache:
        N = vertices.shape[0]
        pos = 3
        normal = 3
        tex_coord = 2
        color = 3
        vertex_stride = pos + normal + tex_coord + color
        vbo = Vector.field(vertex_stride, f32, shape=(N, ))
        vbo_field_cache[vertices] = vbo
        return vbo
    else:
        return vbo_field_cache[vertices]


@kernel
def copy_to_vbo(vbo:template(),src:template(),offset:template(),num_components:template()):
    for i in src:
        vbo[i][offset + 0] = src[i][0]
        vbo[i][offset + 1] = src[i][1]
        if ti.static(num_components == 3):
            vbo[i][offset + 2] = src[i][2]

def validate_input_field(f,name):
    if f.dtype != f32:
        raise Exception(f"{name} needs to have dtype f32")
    if hasattr(f, 'n'):
        if f.m != 1:
            raise Exception(f'{name} needs to be a Vector field (matrix with 1 column)')
    else:
        raise Exception(f'{name} needs to be a Vector field')
    if len(f.shape) != 1:
        raise Exception(f"the shape of {name} needs to be 1-dimensional")
    

def copy_vertices_to_vbo(vbo,vertices):
    validate_input_field(vertices,"vertices")
    if not 2 <= vertices.n <= 3:
        raise Exception(f'vertices can only be 2D or 3D vector fields')
    copy_to_vbo(vbo,vertices,0,vertices.n)

def copy_normals_to_vbo(vbo,normals):
    validate_input_field(normals,"normals")
    if normals.n != 3:
        raise Exception(f'normals can only be 3D vector fields')
    copy_to_vbo(vbo,normals,3,normals.n)

def copy_texcoords_to_vbo(vbo,texcoords):
    validate_input_field(texcoords,"texcoords")
    if texcoords.n != 2:
        raise Exception(f'texcoords can only be 3D vector fields')
    copy_to_vbo(vbo,texcoords,6,texcoords.n)

def copy_colors_to_vbo(vbo,colors):
    validate_input_field(colors,"colors")
    if colors.n != 3:
        raise Exception(f'colors can only be 3D vector fields')
    copy_to_vbo(vbo,colors,8,colors.n)