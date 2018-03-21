from taichi.core import tc_core
from taichi.misc.util import *
import taichi as tc

class Simulation:
  def __init__(self, name, **kwargs):
    res = kwargs['res']
    if len(res) == 2:
      self.c = tc_core.create_simulation2(name)
      self.Vector = tc_core.Vector2f
      self.Vectori = tc_core.Vector2i
    else:
      self.c = tc.core.create_simulation3(name)
      self.Vector = tc_core.Vector3f
      self.Vectori = tc_core.Vector3i


  def __getattr__(self, item):
    if item not in self.__dict__:
      # Goes to general action
      def action(**kwargs):
        print("general action generating")
        return self.c.general_action(name=item, **kwargs)
      return action
