from taichi.util import *
from taichi.core import tc_core

class Camera:
    def __init__(self, name, **kwargs):
        self.c = tc_core.create_camera(name)
        self.c.initialize(config_from_dict(kwargs))

