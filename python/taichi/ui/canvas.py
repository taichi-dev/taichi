from .staging_buffer import (copy_colors_to_vbo, copy_vertices_to_vbo,
                             get_vbo_field, to_u8_rgba)
from .utils import get_field_info


class Canvas:
    def __init__(self, canvas) -> None:
        self.canvas = canvas  #reference to a PyCanvas

    def set_background_color(self, color):
        self.canvas.set_background_color(color)

    def set_image(self, img):
        staging_img = to_u8_rgba(img)
        info = get_field_info(staging_img)
        self.canvas.set_image(info)

    def triangles(self,
                  vertices,
                  color=(0.5, 0.5, 0.5),
                  indices=None,
                  per_vertex_color=None):
        """Declare a set of 2D triangles inside the scene.

        Args:
            vertices: a taichi 2D Vector field, where each element indicate the 3D location of a vertex.
            indices: a taichi int field of shape (3 * #triangles), which indicate the vertex indices of the triangles. If this is None, then it is assumed that the vertices are already arranged in triangles order.
            color: a global color for the triangles as 3 floats representing RGB values. If `per_vertex_color` is provided, this is ignored.
            per_vertex_color (Tuple[float]): a taichi 3D vector field, where each element indicate the RGB color of a vertex.
        """
        vbo = get_vbo_field(vertices)
        copy_vertices_to_vbo(vbo, vertices)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        vbo_info = get_field_info(vbo)
        indices_info = get_field_info(indices)
        self.canvas.triangles(vbo_info, indices_info, has_per_vertex_color,
                              color)

    def lines(self,
              vertices,
              width,
              indices=None,
              color=(0.5, 0.5, 0.5),
              per_vertex_color=None):
        """Declare a set of 2D lines inside the scene.

        Args:
            vertices: a taichi 2D Vector field, where each element indicate the 3D location of a vertex.
            width (float): width of the lines, relative to the height of the screen.
            indices: a taichi int field of shape (2 * #lines), which indicate the vertex indices of the lines. If this is None, then it is assumed that the vertices are already arranged in lines order.
            color: a global color for the triangles as 3 floats representing RGB values. If `per_vertex_color` is provided, this is ignored.
            per_vertex_color (Tuple[float]): a taichi 3D vector field, where each element indicate the RGB color of a vertex.
        """
        vbo = get_vbo_field(vertices)
        copy_vertices_to_vbo(vbo, vertices)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        vbo_info = get_field_info(vbo)
        indices_info = get_field_info(indices)
        self.canvas.lines(vbo_info, indices_info, has_per_vertex_color, color,
                          width)

    def circles(self,
                centers,
                radius,
                color=(0.5, 0.5, 0.5),
                per_vertex_color=None):
        """Declare a set of 2D circles inside the scene.

        Args:
            centers: a taichi 2D Vector field, where each element indicate the 3D location of a vertex.
            radius (float): radius of the circles, relative to the height of the screen.
            color: a global color for the triangles as 3 floats representing RGB values. If `per_vertex_color` is provided, this is ignored.
            per_vertex_color (Tuple[float]): a taichi 3D vector field, where each element indicate the RGB color of a circle.
        """
        vbo = get_vbo_field(centers)
        copy_vertices_to_vbo(vbo, centers)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        vbo_info = get_field_info(vbo)
        self.canvas.circles(vbo_info, has_per_vertex_color, color, radius)

    def scene(self, scene):
        """Draw a 3D scene on the canvas"""
        self.canvas.scene(scene)
