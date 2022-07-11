from taichi._lib import core as _ti_core
from taichi.lang.impl import field
from taichi.lang.kernel_impl import kernel
from taichi.lang.matrix import Vector
from taichi.lang.ops import atomic_add
from taichi.types.annotations import template
from taichi.types.primitive_types import f32

from .staging_buffer import (copy_colors_to_vbo, copy_normals_to_vbo,
                             copy_vertices_to_vbo, get_vbo_field)
from .utils import check_ggui_availability, get_field_info

normals_field_cache = {}


def get_normals_field(vertices):
    if vertices not in normals_field_cache:
        N = vertices.shape[0]
        normals = Vector.field(3, f32, shape=(N, ))
        normal_weights = field(f32, shape=(N, ))
        normals_field_cache[vertices] = (normals, normal_weights)
        return (normals, normal_weights)
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


class Scene:
    """The 3D scene class, which can contain meshes and particles,
    and can be rendered on a canvas.
    """
    def __init__(self):
        check_ggui_availability()
        self.scene = _ti_core.PyScene()

    def set_camera(self, camera):
        """Set the camera for this scene.

        Args:
            camera (:class:`~taichi.ui.Camera`): A camera instance.
        """
        self.scene.set_camera(camera.ptr)

    def mesh(self,
             vertices,
             indices=None,
             normals=None,
             color=(0.5, 0.5, 0.5),
             per_vertex_color=None,
             two_sided=False,
             vertex_offset: int = 0,
             vertex_count: int = None,
             index_offset: int = 0,
             index_count: int = None):
        """Declare a mesh inside the scene.
        if you indicate the index_offset and index_count, the normals will also
        be sliced by the args, and the shading resultes will not be affected.
        (It is equal to make a part of the mesh visible)
        Args:
            vertices: a taichi 3D Vector field, where each element indicate the
                3D location of a vertex.
            indices: a taichi int field of shape (3 * #triangles), which indicate
                the vertex indices of the triangles. If this is None, then it is
                assumed that the vertices are already arranged in triangles order.
            normals: a taichi 3D Vector field, where each element indicate the
                normal of a vertex. If this is none, normals will be automatically
                inferred from vertex positions.
            color: a global color of the mesh as 3 floats representing RGB values.
                If `per_vertex_color` is provided, this is ignored.
            per_vertex_color (Tuple[float]): a taichi 3D vector field, where each
                element indicate the RGB color of a vertex.
            two_sided (bool): whether or not the triangles should be able to be
                seen from both sides.
            vertex_offset: int type(ohterwise float type will be floored to int),
                if 'indices' is provided, this means the value added to the vertex
                index before indexing into the vertex buffer, else this means the
                index of the first vertex to draw.
            vertex_count: int type(ohterwise float type will be floored to int),
                only avaliable when `indices` is not provided, which is the number
                of vertices to draw.
            index_offset: int type(ohterwise float type will be floored to int),
                only avaliable when `indices` is provided, which is the base index
                within the index buffer.
            index_count: int type(ohterwise float type will be floored to int),
                only avaliable when `indices` is provided, which is the the number
                of vertices to draw.
        """
        vbo = get_vbo_field(vertices)
        copy_vertices_to_vbo(vbo, vertices)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        if normals is None:
            normals = gen_normals(vertices, indices)
        if vertex_count is None:
            vertex_count = vertices.shape[0]
        if index_count is None:
            index_count = indices.shape[0]
        copy_normals_to_vbo(vbo, normals)
        vbo_info = get_field_info(vbo)
        indices_info = get_field_info(indices)

        self.scene.mesh(vbo_info, has_per_vertex_color, indices_info, color,
                        two_sided, index_count, index_offset, vertex_count,
                        vertex_offset)

    def particles(self,
                  centers,
                  radius,
                  color=(0.5, 0.5, 0.5),
                  per_vertex_color=None,
                  index_offset: int = 0,
                  index_count: int = None):
        """Declare a set of particles within the scene.

        Args:
            centers: a taichi 3D Vector field, where each element indicate the
                3D location of the center of a triangle.
            color: a global color for the particles as 3 floats representing RGB
                values. If `per_vertex_color` is provided, this is ignored.
            per_vertex_color (Tuple[float]): a taichi 3D vector field, where each
                element indicate the RGB color of a particle.
            index_offset: int type(ohterwise float type will be floored to int),
                the index of the first vertex to draw.
            index_count: int type(ohterwise float type will be floored to int),
                the number of vertices to draw.
        """
        vbo = get_vbo_field(centers)
        copy_vertices_to_vbo(vbo, centers)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        if index_count is None:
            index_count = centers.shape[0]
        vbo_info = get_field_info(vbo)
        self.scene.particles(vbo_info, has_per_vertex_color, color, radius,
                             index_count, index_offset)

    def point_light(self, pos, color):  # pylint: disable=W0235
        """Set a point light in this scene.

        Args:
            pos (list, tuple, :class:`~taichi.types.vector(3, float)`):
                3D vector for light position.
            color (list, tuple, :class:`~taichi.types.vector(3, float)`):
                (r, g, b) triple for the color of the light, in the range [0, 1].
        """
        self.scene.point_light(pos, color)

    def ambient_light(self, color):
        """Set the ambient color of this scene.

        Example::

            >>> scene = ti.ui.Scene()
            >>> scene.ambient_light([0.2, 0.2, 0.2])
        """
        self.scene.ambient_light(color)
