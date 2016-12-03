from taichi.util import *
from taichi.core import tc_core

class Scene:
    def __init__(self):
        self.c = tc_core.create_scene()

    def add_mesh(self, mesh):
        self.c.add_mesh(mesh.c)

    def __getattr__(self, key):
        return self.c.__getattribute__(key)

    def set_atmosphere_material(self, mat):
        self.c.set_atmosphere_material(mat.c)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
