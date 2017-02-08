from taichi.core import tc_core
from taichi.tools.transform import Transform
from taichi.misc.util import *
import traceback

current_transform = [tc_core.Matrix4(1.0)]


class TransformScope:
    def __init__(self, transform=None, translate=None, rotation=None, scale=None):
        self.transform = Transform(transform, translate, rotation, scale).get_matrix()

    def __enter__(self):
        self.previous_transform = get_current_transform()
        set_current_tranform(self.previous_transform * self.transform)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            traceback.print_exception(exc_type, exc_val, exc_tb)
            exit(-1)
        set_current_tranform(self.previous_transform)


def get_current_transform():
    return current_transform[0]


def set_current_tranform(transform):
    current_transform[0] = transform


transform_scope = TransformScope
