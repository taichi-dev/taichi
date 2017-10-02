from taichi.core import tc_core
from taichi.misc.util import *


class Camera:

  def __init__(self, name, width=None, height=None, **kwargs):
    if width != None:
      kwargs['res'] = (width, height)
      print('Warning: width and height for Camera initialization is deprecated!')
    self.c = tc_core.create_camera(name)
    self.c.initialize(config_from_dict(kwargs))
