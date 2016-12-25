import taichi as tc
from taichi.util import *
import math


def create_mesh_from_functions(res, surface, normal=None, uv=None):
    surface = tc.core.function23_from_py_obj(surface)
    if normal:
        normal = tc.core.function23_from_py_obj(normal)
    else:
        normal = None
    if uv:
        uv = tc.core.function22_from_py_obj
    else:
        uv = None
    return tc.core.generate_mesh(Vectori(res), surface, normal, uv)


def create_sphere(res):
    res = Vectori(res)

    def surface(uv):
        theta = uv.x * math.pi * 2
        phi = -uv.y * math.pi
        return Vector(math.cos(theta) * math.sin(phi), math.cos(phi), math.sin(theta) * math.sin(phi))

    # norm = surf
    return create_mesh_from_functions(res, surface, surface)


def create_plane(res):
    res = Vectori(res)

    def surface(uv):
        return Vector(uv.x * 2 - 1, 0, -uv.y * 2 + 1)

    return create_mesh_from_functions(res, surface)


def rotate_y(v, r):
    c, s = math.cos(r), math.sin(r)
    return Vector(c * v.x + s * v.z, v.y, -s * v.x + c * v.z)


def create_torus(res, inner=0.5, outer=1.0):
    res = Vectori(res)

    def surface(uv):
        theta = uv.x * math.pi * 2
        phi = uv.y * math.pi * 2
        center = (inner + outer) / 2
        radius = outer - center
        return rotate_y(Vector(center + radius * math.cos(phi), radius * math.sin(phi), 0), theta)

    return create_mesh_from_functions(res, surface)
