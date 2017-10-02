import taichi.core as tc_core
from taichi.misc.util import Vector


class Transform:

  def __init__(self,
               transform=tc_core.Matrix4(1.0),
               translate=(0, 0, 0),
               rotation=(0, 0, 0),
               scale=(1, 1, 1)):
    if transform is None:
      transform = tc_core.Matrix4(1.0)
    self.transform = transform
    if scale is not None:
      self.scale(scale)
    if rotation is not None:
      self.rotate(rotation)
    if translate is not None:
      self.translate(translate)

  def translate(self, trans):
    self.transform = self.transform.translate(Vector(trans))

  def rotate(self, rot):
    self.transform = self.transform.rotate_euler(Vector(rot))

  def scale(self, scale):
    if isinstance(scale, tuple):
      self.transform = self.transform.scale(Vector(scale))
    else:
      self.transform = self.transform.scale_s(scale)

  def get_matrix(self):
    return self.transform
