import math

import taichi as tc
from taichi.misc.util import *


def create_mesh_from_functions(res, surface, normal=None, uv=None, smooth=True):
  surface = tc.core.function23_from_py_obj(surface)
  if normal:
    normal = tc.core.function23_from_py_obj(normal)
  else:
    normal = None
  if uv:
    uv = tc.core.function22_from_py_obj
  else:
    uv = None
  return tc.core.generate_mesh(Vectori(res), surface, normal, uv, smooth)


def create_sphere(res=(60, 60), smooth=True):
  res = Vectori(res)

  def surface(uv):
    theta = uv.x * math.pi * 2
    phi = -uv.y * math.pi
    return Vector(
        math.cos(theta) * math.sin(phi),
        math.cos(phi), math.sin(theta) * math.sin(phi))

  # norm = surf
  return create_mesh_from_functions(res, surface, surface, smooth=smooth)


def create_plane(res=(1, 1)):
  res = Vectori(res)

  def surface(uv):
    return Vector(uv.x * 2 - 1, 0, -uv.y * 2 + 1)

  return create_mesh_from_functions(res, surface)


def rotate_y(v, r):
  c, s = math.cos(r), math.sin(r)
  return Vector(c * v.x + s * v.z, v.y, -s * v.x + c * v.z)


def create_torus(res=(100, 30), inner=0.5, outer=1.0, smooth=True):
  res = Vectori(res)

  def surface(uv):
    theta = uv.x * math.pi * 2
    phi = uv.y * math.pi * 2
    center = (inner + outer) / 2
    radius = outer - center
    return rotate_y(
        Vector(center + radius * math.cos(phi), radius * math.sin(phi), 0),
        theta)

  return create_mesh_from_functions(res, surface, smooth=smooth)


def create_mobius(res, radius, width, loops=1, smooth=True):
  res = Vectori(res)

  def surface(uv):
    theta = uv.x * math.pi * 2
    t = (uv.y - 0.5) * width
    phi = theta * loops
    return rotate_y(
        Vector(radius + t * math.cos(phi), t * math.sin(phi), 0), theta)

  return create_mesh_from_functions(res, surface, smooth=smooth)


def create_merged(a, b):
  return tc.core.merge_mesh(a, b)


def create_cone(res, smooth=True):

  def surface(uv):
    theta = uv.x * math.pi * 2
    return rotate_y((1 - uv.y) * Vector(1, -1, 0) + uv.y * Vector(0, 1, 0),
                    theta)

  def normal(uv):
    return rotate_y(Vector(2, 1, 0), uv.x * 2 * math.pi)

  body = create_mesh_from_functions(res, surface, normal, smooth=smooth)
  cap = create_mesh_from_functions(
      res, lambda uv: rotate_y(Vector(uv.y, -1, 0), -2 * math.pi * uv.x),
      lambda uv: Vector(0, 1, 0))
  return create_merged(cap, body)


def create_cylinder(res, smooth=True):

  def surface(uv):
    theta = uv.x * math.pi * 2
    return rotate_y((1 - uv.y) * Vector(1, -1, 0) + uv.y * Vector(1, 1, 0),
                    theta)

  body = create_mesh_from_functions(res, surface, smooth=smooth)
  cap2 = create_mesh_from_functions(
      res, lambda uv: rotate_y(Vector(uv.y, 1, 0), 2 * math.pi * uv.x),
      lambda uv: Vector(0, -1, 0))
  cap1 = create_mesh_from_functions(
      res, lambda uv: rotate_y(Vector(uv.y, -1, 0), -2 * math.pi * uv.x),
      lambda uv: Vector(0, 1, 0))
  cap = create_merged(cap1, cap2)
  return create_merged(cap, body)

class SegmentMesh:
  def __init__(self):
    self.segments = []
  
  def add_segment(self, segment):
    assert(len(segment) == 2)
    assert(isinstance(segment[0], tuple))
    assert(isinstance(segment[1], tuple))
    self.segments.append(segment)
  
  def __str__(self):
    ret = ''
    ret += '{} '.format(len(self.segments))
    for seg in self.segments:
      ret += '{} {} {} {} '.format(seg[0][0], seg[0][1], seg[1][0], seg[1][1])
    return ret

