from taichi.core import tc_core
from taichi.misc.util import *
import taichi as tc

class Simulation:
  def __init__(self, **kwargs):
    res = kwargs['res']
    name = kwargs['name']
    self.res = res
    if len(res) == 2:
      self.c = tc_core.create_simulation2(name)
      self.Vector = tc_core.Vector2f
      self.Vectori = tc_core.Vector2i
    else:
      self.c = tc.core.create_simulation3(name)
      self.Vector = tc_core.Vector3f
      self.Vectori = tc_core.Vector3i
    self.c.initialize(config_from_dict(kwargs))

  def general_action(self, action, **kwargs):
    assert 'action' not in kwargs
    kwargs['action'] = action
    return self.c.general_action(config_from_dict(kwargs))

  def __getattr__(self, item):
    if item not in self.__dict__:
      # Goes to general action
      def action(**kwargs):
        return self.general_action(item, **kwargs)
      return action
