from taichi.util import *
from taichi.core import tc_core

class VolumeMaterial:
    def __init__(self, name, **kwargs):
        self.c = tc_core.create_volume_material(name)
        self.c.initialize(config_from_dict(kwargs))
