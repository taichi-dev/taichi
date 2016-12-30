from taichi.core import tc_core
from taichi.mics.util import *


class Camera:
    def __init__(self, name, **kwargs):
        self.c = tc_core.create_camera(name)
        self.c.initialize(config_from_dict(kwargs))

