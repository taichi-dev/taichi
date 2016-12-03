from taichi.util import *
from taichi.core import tc_core

class Mesh:
    def __init__(self, filename, material):
        self.c = tc_core.create_mesh()
        self.c.initialize(config_from_dict({'filename': filename}))
        self.c.set_material(material.c)

    def __getattr__(self, key):
        return self.c.__getattribute__(key)
