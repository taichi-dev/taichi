from taichi.core import tc_core
from taichi.core import unit
from taichi.misc.util import *


@unit('envmap')
class EnvironmentMap:

  def set_transform(self, transform):
    self.c.set_transform(transform)
