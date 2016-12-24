from taichi.core import tc_core

current_transform = [tc_core.Matrix4(1.0)]

class TransformScope:
    def __init__(self, transform):
        self.transform = transform

def get_current_transform():
    return current_transform[0]

