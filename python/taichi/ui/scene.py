import pathlib

from taichi.core import ti_core as _ti_core
from taichi.core.primitive_types import f32
from taichi.lang.impl import default_cfg, field
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel
from taichi.lang.matrix import Vector
from taichi.lang.ops import atomic_add, get_addr

from .camera import Camera
from .staging_buffer import (copy_colors_to_vbo, copy_normals_to_vbo,
                             copy_vertices_to_vbo, get_vbo_field)
from .utils import get_field_info

normals_field_cache = {}


def get_normals_field(vertices):
    if vertices not in normals_field_cache:
        N = vertices.shape[0]
        normals = Vector.field(3, f32, shape=(N, ))
        normal_weights = field(f32, shape=(N, ))
        normals_field_cache[vertices] = (normals, normal_weights)
        return (normals, normal_weights)
    else:
        return normals_field_cache[vertices]


@kernel
def gen_normals_kernel(vertices: template(), normals: template()):
    N = vertices.shape[0]
    for i in range(N / 3):
        a = vertices[i * 3]
        b = vertices[i * 3 + 1]
        c = vertices[i * 3 + 2]
        n = (a - b).cross(a - c).normalized()
        normals[i * 3] = n
        normals[i * 3 + 1] = n
        normals[i * 3 + 2] = n


@kernel
def gen_normals_kernel_indexed(vertices: template(), indices: template(),
                               normals: template(), weights: template()):
    num_triangles = indices.shape[0] / 3
    num_vertices = vertices.shape[0]
    for i in range(num_vertices):
        normals[i] = Vector([0.0, 0.0, 0.0])
        weights[i] = 0.0
    for i in range(num_triangles):
        i_a = indices[i * 3]
        i_b = indices[i * 3 + 1]
        i_c = indices[i * 3 + 2]
        a = vertices[i_a]
        b = vertices[i_b]
        c = vertices[i_c]
        n = (a - b).cross(a - c).normalized()
        atomic_add(normals[i_a], n)
        atomic_add(normals[i_b], n)
        atomic_add(normals[i_c], n)
        atomic_add(weights[i_a], 1.0)
        atomic_add(weights[i_b], 1.0)
        atomic_add(weights[i_c], 1.0)
    for i in range(num_vertices):
        if weights[i] > 0.0:
            normals[i] = normals[i] / weights[i]


def gen_normals(vertices, indices):
    normals, weights = get_normals_field(vertices)
    if indices is None:
        gen_normals_kernel(vertices, normals)
    else:
        gen_normals_kernel_indexed(vertices, indices, normals, weights)
    return normals


class Scene(_ti_core.PyScene):
    def __init__(self):
        super().__init__()

    def set_camera(self, camera):
        super().set_camera(camera.ptr)

    def mesh(self,
             vertices,
             indices=None,
             normals=None,
             color=(0.5, 0.5, 0.5),
             per_vertex_color=None,
             two_sided=False):
        vbo = get_vbo_field(vertices)
        copy_vertices_to_vbo(vbo, vertices)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        if normals is None:
            normals = gen_normals(vertices, indices)
        copy_normals_to_vbo(vbo, normals)
        vbo_info = get_field_info(vbo)
        indices_info = get_field_info(indices)

        super().mesh(vbo_info, has_per_vertex_color, indices_info, color,
                     two_sided)

    def particles(self,
                  vertices,
                  radius,
                  color=(0.5, 0.5, 0.5),
                  per_vertex_color=None):
        vbo = get_vbo_field(vertices)
        copy_vertices_to_vbo(vbo, vertices)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        vbo_info = get_field_info(vbo)
        super().particles(vbo_info, has_per_vertex_color, color, radius)

    def point_light(self, pos, color):
        super().point_light(pos, color)

    def ambient_light(self, color):
        super().ambient_light(color)
