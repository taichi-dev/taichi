import taichi as tc
from taichi.misc.util import *


class LevelSet:

  def __init__(self, res, offset=None):
    if len(res) == 2:
      self.Vector = tc.core.Vector2f
      self.Vectori = tc.core.Vector2i
    else:
      self.Vector = tc.core.Vector3f
      self.Vectori = tc.core.Vector3i

    if offset is None:
      offset = self.Vector(0.5)
    self.delta_x = 1.0 / res.min()
    print(self.delta_x)
    self.res = res + self.Vectori(1)
    if len(res) == 2:
      self.levelset = tc.core.LevelSet2D(self.res, offset)
      self.id = tc.core.register_levelset2d(self.levelset)
    else:
      self.levelset = tc.core.LevelSet3D(self.res, offset)
      self.id = tc.core.register_levelset3d(self.levelset)

  def add_sphere(self, center, radius, inside_out=False):
    if type(center) not in [tc.core.Vector2f, tc.core.Vector3f]:
      center = Vector(center)
    self.levelset.add_sphere(
        Vector(center) * (1.0 / self.delta_x), radius / self.delta_x,
        inside_out)

  def add_polygon(self, vertices, inside_out=False):
    self.levelset.add_polygon(make_polygon(vertices, 1 / self.delta_x), inside_out)

  def add_plane(self, normal, d):
    self.levelset.add_plane(normal, d / self.delta_x)

  def add_cuboid(self, lower_boundry, upper_boundry, inside_out=False):
    self.levelset.add_cuboid(
        Vector(lower_boundry[0] / self.delta_x, lower_boundry[1] / self.delta_x,
               lower_boundry[2] / self.delta_x),
        Vector(upper_boundry[0] / self.delta_x, upper_boundry[1] / self.delta_x,
               upper_boundry[2] / self.delta_x), inside_out)

  def global_increase(self, delta):
    self.levelset.global_increase(delta / self.delta_x)

  def set_friction(self, f):
    self.levelset.friction = f

  def get_delta_x(self):
    return self.delta_x
