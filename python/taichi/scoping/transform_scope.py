from taichi.core import tc_core
from taichi.misc.util import *

current_transform = [tc_core.Matrix4(1.0)]

class TransformScope:
    def __init__(self, transform=tc_core.Matrix4(1.0), translate=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        transform = transform.scale(Vector(scale)).rotate_euler(Vector(rotation)).translate(Vector(translate))
        self.transform = transform

    def __enter__(self):
        self.previous_transform = get_current_transform()
        set_current_tranform(self.previous_transform * self.transform)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            raise exc_val
        set_current_tranform(self.previous_transform)

def get_current_transform():
    return current_transform[0]

def set_current_tranform(transform):
    current_transform[0] = transform

transform_scope = TransformScope
