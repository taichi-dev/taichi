from taichi.util import *
from taichi.core import tc_core

def map_filename(name):
    if name.rfind('/') == -1:
        filename = '../assets/meshes/%s.obj' % name
    else:
        filename = name
    return filename

class Mesh:
    def __init__(self, filename, material, translate=Vector(0, 0, 0), rotation=Vector(0, 0, 0), scale=Vector(1, 1, 1),
                 transform=None):
        filename = map_filename(filename)
        self.c = tc_core.create_mesh()
        self.c.initialize(config_from_dict({'filename': filename}))
        if transform:
            self.c.transform = transform
        self.c.set_material(material.c)
        self.scale(scale)
        self.rotate_euler(rotation)
        self.translate(translate)

    def set_transform(self, transform):
        self.c.transform = transform

    def scale(self, s):
        if isinstance(s, float) or isinstance(s, int):
            self.c.transform = self.c.transform.scale_s(float(s))
        else:
            self.c.transform = self.c.transform.scale(Vector(s))

    def rotate_euler(self, rotation):
        self.c.transform = self.c.transform.rotate_euler(Vector(rotation))

    def translate(self, translate):
        self.c.transform = self.c.transform.translate(Vector(translate))

    def __getattr__(self, key):
        return self.c.__getattribute__(key)
