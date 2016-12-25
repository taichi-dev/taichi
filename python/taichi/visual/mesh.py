from taichi.util import *
from taichi.core import tc_core
import taichi.geometry as geometry
from taichi.scoping.transform_scope import get_current_transform


def map_filename(name):
    if name == 'plane':
        return geometry.create_plane((1, 1))
    elif name == 'sphere':
        return geometry.create_sphere((100, 100))
    elif name == 'torus':
        return geometry.create_torus((100, 100), 0.7, 1.0)
    if name.rfind('/') == -1:
        filename = '../assets/meshes/%s.obj' % name
    else:
        filename = geometry
    return filename


class Mesh:
    def __init__(self, filename_or_triangles, material=None, translate=Vector(0, 0, 0), rotation=Vector(0, 0, 0),
                 scale=Vector(1, 1, 1),
                 transform=None):
        if isinstance(filename_or_triangles, str):
            filename_or_triangles = map_filename(filename_or_triangles)
        self.c = tc_core.create_mesh()
        if isinstance(filename_or_triangles, str):
            self.c.initialize(config_from_dict({'filename': filename_or_triangles}))
        else:
            self.c.initialize(config_from_dict({'filename': ''}))
            self.c.set_untransformed_triangles(filename_or_triangles)
        if transform:
            self.c.transform = transform
        self.c.set_material(material.c)
        self.scale(scale)
        self.rotate_euler(rotation)
        self.translate(translate)
        self.set_transform(get_current_transform() * self.c.transform)

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
