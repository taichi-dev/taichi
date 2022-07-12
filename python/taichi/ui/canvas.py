from taichi.lang.util import warning

from .staging_buffer import (copy_colors_to_vbo, copy_vertices_to_vbo,
                             get_vbo_field, to_u8_rgba)
from .utils import get_field_info


def _clamp_color_component(component):
    if isinstance(component, int):
        return max(min(component, 255), 0.0) / 255.0
    elif isinstance(component, float):
        return max(min(float(component), 1.0), 0.0)
    else:
        return None


def _translate_color(color):
    COLOR_LUT = {
        # See https://matplotlib.org/stable/gallery/color/named_colors.html.
        'r': (1.0, 0.0, 0.0),
        'g': (0.0, 1.0, 0.0),
        'b': (0.0, 0.0, 1.0),
        'c': (0.0, 0.75, 0.75),
        'm': (0.75, 0.0, 0.75),
        'y': (0.75, 0.75, 0.0),
        'k': (0.0, 0.0, 0.0),
        'w': (1.0, 1.0, 1.0),
        'blue': '#1f77b4',
        'orange': '#ff7f0e',
        'green': '#2ca02c',
        'red': '#d62728',
        'purple': '#9467bd',
        'brown': '#8c564b',
        'pink': '#e377c2',
        'gray': '#7f7f7f',
        'olive': '#bcbd22',
        'cyan': '#17becf',
    }
    ERROR_COLOR = (1.0, 0.0, 1.0)

    if isinstance(color, str):
        if color.startswith('#'):
            # CSS hexadecimal representation.
            color = color[1:]
            if len(color) == 6 and color:
                r = int(color[0:2], 16)
                g = int(color[2:4], 16)
                b = int(color[4:6], 16)
                return _translate_color((r, g, b))
            elif len(color) == 3 and color:
                r = int(color[0], 16)
                g = int(color[1], 16)
                b = int(color[2], 16)
                return _translate_color((r * 16 + r, g * 16 + g, b * 16 + b))
        else:
            if color in COLOR_LUT:
                return _translate_color(COLOR_LUT[color])
    elif isinstance(color, tuple) or isinstance(color, list):
        color = list(color[:3]) + [0] * (3 - len(color))
        if all(isinstance(x, int) and x > 1 for x in color):
            return tuple(_clamp_color_component(x) for x in color)
        elif all(isinstance(x, int) or isinstance(x, float) for x in color):
            return tuple(_clamp_color_component(float(x)) for x in color)

    warning(f"'{color}' is not a valid color")
    return ERROR_COLOR


class Canvas:
    """The Canvas class.

    This is the context manager for managing drawing commands on a window.
    You should not instantiate this class directly via `__init__`, instead
    please call the `get_canvas()` method of :class:`~taichi.ui.Window`.
    """
    def __init__(self, canvas) -> None:
        self.canvas = canvas  #reference to a PyCanvas

    def set_background_color(self, color):
        """Set the background color of this canvas.

        Args:
            color (tuple(float)): RGB triple in the range [0, 1].
        """
        self.canvas.set_background_color(_translate_color(color))

    def set_image(self, img):
        """Set the content of this canvas to an `img`.

        Args:
            img (numpy.ndarray, :class:`~taichi.MatrixField`, :class:`~taichi.Field`): \
                the image to be shown.
        """
        staging_img = to_u8_rgba(img)
        info = get_field_info(staging_img)
        self.canvas.set_image(info)

    def triangles(self,
                  vertices,
                  color=(0.5, 0.5, 0.5),
                  indices=None,
                  per_vertex_color=None):
        """Draw a set of 2D triangles on this canvas.

        Args:
            vertices: a taichi 2D Vector field, where each element indicate \
                the 3D location of a vertex.
            indices: a taichi int field of shape (3 * #triangles), which \
                indicate the vertex indices of the triangles. If this is None, \
                then it is assumed that the vertices are already arranged in \
                triangles order.
            color: a global color for the triangles as 3 floats representing \
                RGB values. If `per_vertex_color` is provided, this is ignored.
            per_vertex_color (Tuple[float]): a taichi 3D vector field, \
                where each element indicate the RGB color of a vertex.
        """
        vbo = get_vbo_field(vertices)
        copy_vertices_to_vbo(vbo, vertices)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        vbo_info = get_field_info(vbo)
        indices_info = get_field_info(indices)
        self.canvas.triangles(vbo_info, indices_info, has_per_vertex_color,
                              _translate_color(color))

    def lines(self,
              vertices,
              width,
              indices=None,
              color=(0.5, 0.5, 0.5),
              per_vertex_color=None):
        """Draw a set of 2D lines on this canvas.

        Args:
            vertices: a taichi 2D Vector field, where each element indicate the \
                3D location of a vertex.
            width (float): width of the lines, relative to the height of the screen.
            indices: a taichi int field of shape (2 * #lines), which indicate \
                the vertex indices of the lines. If this is None, then it is \
                assumed that the vertices are already arranged in lines order.
            color: a global color for the triangles as 3 floats representing \
                RGB values. If `per_vertex_color` is provided, this is ignored.
            per_vertex_color (tuple[float]): a taichi 3D vector field, where \
                each element indicate the RGB color of a vertex.
        """
        vbo = get_vbo_field(vertices)
        copy_vertices_to_vbo(vbo, vertices)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        vbo_info = get_field_info(vbo)
        indices_info = get_field_info(indices)
        self.canvas.lines(vbo_info, indices_info, has_per_vertex_color,
                          _translate_color(color), width)

    def circles(self,
                centers,
                radius,
                color=(0.5, 0.5, 0.5),
                per_vertex_color=None):
        """Draw a set of 2D circles on this canvas.

        Args:
            centers: a taichi 2D Vector field, where each element indicate the \
                3D location of a vertex.
            radius (float): radius of the circles, relative to the height of the screen.
            color: a global color for the triangles as 3 floats representing \
                RGB values. If `per_vertex_color` is provided, this is ignored.
            per_vertex_color (Tuple[float]): a taichi 3D vector field, where \
                each element indicate the RGB color of a circle.
        """
        vbo = get_vbo_field(centers)
        copy_vertices_to_vbo(vbo, centers)
        has_per_vertex_color = per_vertex_color is not None
        if has_per_vertex_color:
            copy_colors_to_vbo(vbo, per_vertex_color)
        vbo_info = get_field_info(vbo)
        self.canvas.circles(vbo_info, has_per_vertex_color,
                            _translate_color(color), radius)

    def scene(self, scene):
        """Draw a 3D scene on the canvas

        Args:
            scene (:class:`~taichi.ui.Scene`): an instance of :class:`~taichi.ui.Scene`.
        """
        self.canvas.scene(scene.scene)
