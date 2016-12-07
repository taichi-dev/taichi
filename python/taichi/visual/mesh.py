from taichi.util import *
from taichi.core import tc_core

def map_filename(name):
    if name.rfind('/') == -1:
        filename = '../assets/meshes/%s.obj' % name
    else:
        filename = name
    return filename

class Mesh:
    def __init__(self, filename, material, translate=Vector(0, 0, 0), rotation=Vector(0, 0, 0), scale=Vector(1, 1, 1)):
        filename = map_filename(filename)
        self.c = tc_core.create_mesh()
        self.c.initialize(config_from_dict({'filename': filename}))
        self.c.set_material(material.c)
        self.c.translate(Vector(translate))
        self.c.rotate_euler(Vector(rotation))
        self.scale(scale)

    def scale(self, s):
        if isinstance(s, float) or isinstance(s, int):
            self.c.scale_s(float(s))
        else:
            self.c.scale(Vector(s))

    def __getattr__(self, key):
        return self.c.__getattribute__(key)
