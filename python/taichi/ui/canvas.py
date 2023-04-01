from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang._texture import Texture

from .staging_buffer import (copy_all_to_vbo, get_indices_field, get_vbo_field,
                             to_rgba8)
from .utils import get_field_info


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
        self.canvas.set_background_color(color)

    def set_image(self, img):
        """Set the content of this canvas to an `img`.

        Args:
            img (numpy.ndarray, :class:`~taichi.MatrixField`, :class:`~taichi.Field`, :class:`~taichi.Texture`): \
                the image to be shown.
        """
        is_texture = isinstance(img, Texture)
        prog_is_vk = impl.pytaichi.prog.config().arch == _ti_core.Arch.vulkan
        # FIXME: Remove this hack. Maybe add a query function for whether the texture can be presented
        if is_texture and prog_is_vk:
            self.canvas.set_image_texture(img.tex)
        else:
            staging_img = to_rgba8(img)
            info = get_field_info(staging_img)
            self.canvas.set_image(info)

    def contour(self, scalar_field, cmap_name='plasma', normalize=False):
        """Plot a contour view of a scalar field.

        The input scalar_field will be converted to a Numpy array first, and then plotted
        using Matplotlib's colormap. Users can specify the color map through the cmap_name
        argument.

        Args:
            scalar_field (ti.field): The scalar field being plotted. Must be 2D.
            cmap_name (str, Optional): The name of the color map in Matplotlib.
            normalize (bool, Optional): Display the normalized scalar field if set to True.
            Default is False.
        """
        try:
            import numpy as np  # pylint: disable=import-outside-toplevel
            from matplotlib import \
                cm  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise RuntimeError('Failed to import Numpy and Matplotlib. /\
            Please install Numpy and Matplotlib before using contour().')

        scalar_field_np = scalar_field.to_numpy()
        field_shape = scalar_field_np.shape
        ndim = len(field_shape)
        if ndim != 2:
            raise ValueError(
                'contour() can only be used on a 2D scalar field.')

        if normalize:
            scalar_max = np.max(scalar_field_np)
            scalar_min = np.min(scalar_field_np)
            scalar_field_np = (scalar_field_np - scalar_min) / (scalar_max -
                                                                scalar_min)

        cmap = cm.get_cmap(cmap_name)
        output_rgba = cmap(scalar_field_np)
        output_rgb = output_rgba.astype(np.float32)[:, :, :3]
        output = np.ascontiguousarray(output_rgb)
        self.set_image(output)

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
        has_per_vertex_color = per_vertex_color is not None
        copy_all_to_vbo(vbo, vertices, 0, 0,
                        per_vertex_color if has_per_vertex_color else 0)
        vbo_info = get_field_info(vbo)
        indices_ndarray = None
        if indices:
            indices_ndarray = get_indices_field(indices)
        indices_info = get_field_info(indices_ndarray)
        self.canvas.triangles(vbo_info, indices_info, has_per_vertex_color,
                              color)

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
        has_per_vertex_color = per_vertex_color is not None
        copy_all_to_vbo(vbo, vertices, 0, 0,
                        per_vertex_color if has_per_vertex_color else 0)
        vbo_info = get_field_info(vbo)
        indices_ndarray = None
        if indices:
            indices_ndarray = get_indices_field(indices)
        indices_info = get_field_info(indices_ndarray)
        self.canvas.lines(vbo_info, indices_info, has_per_vertex_color, color,
                          width)

    def circles(self,
                centers,
                radius,
                color=(0.5, 0.5, 0.5),
                per_vertex_color=None):
        """Draw a set of 2D circles on this canvas.

        Args:
            centers: a taichi 2D Vector field, where each element indicate the \
                3D location of a vertex.
            radius (Number): radius of the circles in pixels.
            color: a global color for the triangles as 3 floats representing \
                RGB values. If `per_vertex_color` is provided, this is ignored.
            per_vertex_color (Tuple[float]): a taichi 3D vector field, where \
                each element indicate the RGB color of a circle.
        """
        vbo = get_vbo_field(centers)
        has_per_vertex_color = per_vertex_color is not None
        copy_all_to_vbo(vbo, centers, 0, 0,
                        per_vertex_color if has_per_vertex_color else 0)
        vbo_info = get_field_info(vbo)
        self.canvas.circles(vbo_info, has_per_vertex_color, color, radius)

    def scene(self, scene):
        """Draw a 3D scene on the canvas

        Args:
            scene (:class:`~taichi.ui.Scene`): an instance of :class:`~taichi.ui.Scene`.
        """
        # FIXME: (penguinliong) Add a point light to ensure the allocation of light source SSBO.
        scene.point_light((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.canvas.scene(scene.scene)
