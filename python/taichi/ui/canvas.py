from taichi.core import ti_core as _ti_core
from taichi.lang.impl import default_cfg
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel
from taichi.lang.ops import get_addr

from .utils import *


class Canvas:
    def __init__(self, canvas) -> None:
        self.canvas = canvas  #reference to a PyCanvas

    def set_background_color(self, color):
        self.canvas.set_background_color(color)

    def set_image(self, img):
        info = get_field_info(img)
        self.canvas.set_image(info)

    def triangles(self,
                  vertices,
                  color=(0.5, 0.5, 0.5),
                  indices=None,
                  per_vertex_color=None):
        vertices_info = get_field_info(vertices)
        indices_info = get_field_info(indices)
        colors_info = get_field_info(per_vertex_color)
        self.canvas.triangles(vertices_info, indices_info, colors_info, color)

    def lines(self,
              vertices,
              width,
              indices=None,
              color=(0.5, 0.5, 0.5),
              per_vertex_color=None):
        vertices_info = get_field_info(vertices)
        indices_info = get_field_info(indices)
        colors_info = get_field_info(per_vertex_color)
        self.canvas.lines(vertices_info, indices_info, colors_info, color,
                          width)

    def circles(self,
                vertices,
                radius,
                color=(0.5, 0.5, 0.5),
                per_vertex_color=None):
        vertices_info = get_field_info(vertices)
        colors_info = get_field_info(per_vertex_color)
        self.canvas.circles(vertices_info, colors_info, color, radius)

    def scene(self, scene):
        self.canvas.scene(scene)
