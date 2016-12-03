from taichi.util import *
from taichi.core import tc_core

class EnvironmentMap:
    def __init__(self, name, **kwargs):
        self.c = tc_core.create_environment_map(name)
        self.c.initialize(config_from_dict(kwargs))
