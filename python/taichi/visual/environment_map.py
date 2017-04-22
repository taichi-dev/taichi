from taichi.core import tc_core
from taichi.core import unit
from taichi.misc.util import *


@unit('envmap')
class EnvironmentMap:
    def __init__(self, name, **kwargs):
        self.c = tc_core.create_envmap(name)
        self.c.initialize(config_from_dict(kwargs))

    def set_transform(self, transform):
        self.c.set_transform(transform)
