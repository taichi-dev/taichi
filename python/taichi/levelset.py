import numpy as np
import pyglet

from taichi_utils import *


class LevelSet2D:
    def __init__(self, width, height, delta_x, offset):
        self.delta_x = delta_x
        self.levelset = tc.LevelSet2D(width, height, offset)
        self.width = width
        self.height = height
        self.cache_image = None

    def add_sphere(self, center, radius, inside_out=False):
        if type(center) != tc.Vector2:
            center = Vector(center[0], center[1])
        self.levelset.add_sphere(Vector(center.x / self.delta_x, center.y / self.delta_x), radius / self.delta_x,
                                 inside_out)

    def add_polygon(self, polygon, inside_out):
        self.levelset.add_polygon(make_polygon(polygon, 1.0 / self.delta_x), inside_out)

    def get(self, x, y=None):
        if y is None:
            y = x.y
            x = x.x
        return self.levelset.sample(x / self.delta_x, y / self.delta_x)

    def get_normalized_gradient(self, p):
        p.x /= self.delta_x
        p.y /= self.delta_x
        return self.levelset.get_normalized_gradient(p)

    def get_image(self, width, height, color_255):
        if self.cache_image is None:
            self.cache_image = array2d_to_image(self.levelset, width, height, color_255)
        return self.cache_image

    def set_friction(self, f):
        self.levelset.friction = f

