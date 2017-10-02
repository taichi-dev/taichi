from taichi.core import tc_core
from taichi.misc.util import *


class ParticleRenderer:

  def __init__(self, name, **kwargs):
    self.c = tc_core.create_particle_renderer(name)
    self.c.initialize(config_from_dict(kwargs))

  def set_camera(self, camera):
    self.c.set_camera(camera.c)

  def __getattr__(self, key):
    return self.c.__getattribute__(key)
