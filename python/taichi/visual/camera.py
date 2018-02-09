import taichi
from taichi.core import tc_core as core
from taichi.misc.util import *

class Camera:
  def __init__(self, name, width=None, height=None, **kwargs):
    if width != None or height != None:
      taichi.error('width and height for Camera initialization is deprecated, use res=(width, height) instead')
    self.c = core.create_camera(name)
    self.c.initialize(config_from_dict(kwargs))
