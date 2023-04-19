import math
import numbers
import os

import numpy as np
import taichi.lang
from taichi._kernels import tensor_to_image, vector_to_fast_image, vector_to_image
from taichi._lib import core as _ti_core
from taichi.lang.field import Field, ScalarField

import taichi as ti


# For window creation and drawing in the original ti.GUI system.
class GUI:
    """Taichi Graphical User Interface class.

    Args:
        name (str, optional): The name of the GUI to be constructed.
            Default is 'Taichi'.
        res (Union[int, List[int]], optional): The resolution of created
            GUI. Default is 512*512. If `res` is scalar, then width will be equal to height.
        background_color (int, optional): The background color of created GUI.
            Default is 0x000000.
        show_gui (bool, optional): Specify whether to render the GUI. Default is True.
        fullscreen (bool, optional): Specify whether to render the GUI in
            fullscreen mode. Default is False.
        fast_gui (bool, optional): Specify whether to use fast gui mode of
            Taichi. Default is False.

    Returns:
        :class:`~taichi.misc.gui.GUI` :The created taichi GUI object.

    """

    class Event:
        """Class for holding a gui event.

        An event is represented by:

        + type (PRESS, MOTION, RELEASE)
        + modifier (modifier keys like ctrl, shift, etc)
        + pos (mouse position)
        + key (event key)
        + delta (for holding mouse wheel)
        """

        def __init__(self):
            self.type = None
            self.modifier = None
            self.pos = None
            self.key = None
            self.delta = None

    # Event keys
    SHIFT = "Shift"
    ALT = "Alt"
    CTRL = "Control"
    ESCAPE = "Escape"
    RETURN = "Return"
    TAB = "Tab"
    BACKSPACE = "BackSpace"
    SPACE = " "
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    CAPSLOCK = "Caps_Lock"
    LMB = "LMB"
    MMB = "MMB"
    RMB = "RMB"
    EXIT = "WMClose"
    WHEEL = "Wheel"
    MOVE = "Motion"

    # Event types
    MOTION = _ti_core.KeyEvent.EType.Move
    PRESS = _ti_core.KeyEvent.EType.Press
    RELEASE = _ti_core.KeyEvent.EType.Release

    def __init__(
        self,
        name="Taichi",
        res=512,
        background_color=0x0,
        show_gui=True,
        fullscreen=False,
        fast_gui=False,
    ):
        show_gui = self.get_bool_environ("TI_GUI_SHOW", show_gui)
        fullscreen = self.get_bool_environ("TI_GUI_FULLSCREEN", fullscreen)
        fast_gui = self.get_bool_environ("TI_GUI_FAST", fast_gui)

        self.name = name
        if isinstance(res, numbers.Number):
            res = (res, res)
        self.res = res
        self.fast_gui = fast_gui
        if fast_gui:
            self.img = np.ascontiguousarray(np.zeros(self.res[0] * self.res[1], dtype=np.uint32))
            fast_buf = self.img.ctypes.data
        else:
            # The GUI canvas uses RGBA for storage, therefore we need NxMx4 for an image.
            self.img = np.ascontiguousarray(np.zeros(self.res + (4,), np.float32))
            fast_buf = 0
        self.core = _ti_core.GUI(name, core_veci(*res), show_gui, fullscreen, fast_gui, fast_buf)
        self.canvas = self.core.get_canvas()
        self.background_color = background_color
        self.key_pressed = set()
        self.event = None
        self.frame = 0
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, e_type, val, tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Close this GUI.

        Example::

            >>> while gui.running:
            >>>     if gui.get_event(gui.PRESS, ti.GUI.ESCAPE):
            >>>         gui.close()
            >>>     gui.show()
        """
        self.core = None  # dereference to call GUI::~GUI()

    # Widget system

    class WidgetValue:
        """Class for maintaining id of gui widgets."""

        def __init__(self, gui, wid):
            self.gui = gui
            self.wid = wid

        @property
        def value(self):
            return self.gui.core.get_widget_value(self.wid)

        @value.setter
        def value(self, value):
            self.gui.core.set_widget_value(self.wid, value)

    @staticmethod
    def get_bool_environ(key, default):
        """Get an environment variable and cast it to `bool`.

        Args:
            key (str): The environment variable key.
            default (bool): The default value.
        Return:
            The environment variable value cast to bool. \
            If the value is not found, directly return argument 'default'.
        """
        if key not in os.environ:
            return default
        return bool(int(os.environ[key]))

    def slider(self, text, minimum, maximum, step=1):
        """Creates a slider object on canvas to be manipulated with.

        Args:
            text (str): The title of slider.
            minimum (int, float): The minimum value of slider.
            maximum (int, float): The maximum value of slider.
            step (int, float): The changing step of slider. Optional and default to 1.

        Return:
            :class:`~taichi.misc.gui.GUI.WidgetValue` :The created slider object.
        """
        wid = self.core.make_slider(text, minimum, minimum, maximum, step)
        return GUI.WidgetValue(self, wid)

    def label(self, text):
        """Creates a label object on canvas.

        Args:
            text (str): The title of label.

        Return:
            :class:`~taichi.misc.gui.GUI.WidgetValue` :The created label object.

        """
        wid = self.core.make_label(text, 0)
        return GUI.WidgetValue(self, wid)

    def button(self, text, event_name=None):
        """Create a button object on canvas to be manipulated with.

        Args:
            text (str): The title of button.
            event_name (str, optional): The event name associated with button.
                Default is WidgetButton_{text}

        Return:
            The event name associated with created button.

        """
        event_name = event_name or f"WidgetButton_{text}"
        self.core.make_button(text, event_name)
        return event_name

    # Drawing system

    def clear(self, color=None):
        """Clears the canvas with the color provided.

        Args:
            color (int, optional): Specify the color to clear the canvas. Default
                is the background color of GUI.

        """
        if color is None:
            color = self.background_color
        self.canvas.clear(color)

    def cook_image(self, img):
        """Converts an img to range [0, 1] for display.

        The input image is stored in a `numpy.ndarray`, if it's dtype
        is `int` it will be rescaled and mapped into range [0, 1]. If
        the dtype is `float` it will be directly casted to 32-bit float type.
        """
        if img.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            img = img.astype(np.float32) * (1 / np.iinfo(img.dtype).max)
        elif img.dtype in [np.float16, np.float32, np.float64]:
            img = img.astype(np.float32)
        else:
            raise ValueError(f"Data type {img.dtype} not supported in GUI.set_image")

        if len(img.shape) == 2:
            img = img[..., None]

        if img.shape[2] == 1:
            img = img + np.zeros((1, 1, 4), np.float32)
        if img.shape[2] == 3:
            zeros = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
            img = np.concatenate([img, zeros], axis=2)
        if img.shape[2] == 2:
            zeros = np.zeros((img.shape[0], img.shape[1], 2), np.float32)
            img = np.concatenate([img, zeros], axis=2)

        assert img.shape[2] == 4, "Image must be grayscale, RG, RGB or RGBA"

        res = img.shape[:2]
        assert res == self.res, "Image resolution does not match GUI resolution"
        return np.ascontiguousarray(img)

    def get_image(self):
        """Return the window content as an `numpy.ndarray`.

        Returns:
            :class:`numpy.array` :The image data in numpy contiguous array type.
        """
        self.img = np.ascontiguousarray(self.img)
        self.core.get_img(self.img.ctypes.data)
        return self.img

    def set_image(self, img):
        """Sets an image to display on the window.

        The image pixels are set from the values of `img[i, j]`, where `i` indicates
        the horizontal coordinates (from left to right) and `j` the vertical coordinates
        (from bottom to top).

        If the window size is `(x, y)`, then `img` must be one of:
            - `ti.field(shape=(x, y))`, a gray-scale image
            - `ti.field(shape=(x, y, 3))`, where `3` is for `(r, g, b)` channels
            - `ti.field(shape=(x, y, 2))`, where `2` is for `(r, g)` channels
            - `ti.Vector.field(3, shape=(x, y))` `(r, g, b)` channels on each component
            - `ti.Vector.field(2, shape=(x, y))` `(r, g)` channels on each component
            - `np.ndarray(shape=(x, y))`
            - `np.ndarray(shape=(x, y, 3))`
            - `np.ndarray(shape=(x, y, 2))`

        The data type of `img` must be one of:
            - `uint8`, range `[0, 255]`
            - `uint16`, range `[0, 65535]`
            - `uint32`, range `[0, 4294967295]`
            - `float32`, range `[0, 1]`
            - `float64`, range `[0, 1]`

        Args:
            img (Union[:class:`taichi.field`, `numpy.array`]): The color array \
                representing the image to be drawn. Support greyscale, RG, RGB, \
                and RGBA color representations. Its shape must match GUI resolution.
        """

        if self.fast_gui:
            assert isinstance(
                img, taichi.lang.matrix.MatrixField
            ), "Only ti.Vector.field is supported in GUI.set_image when fast_gui=True"
            assert img.shape == self.res, "Image resolution does not match GUI resolution"
            assert img.n in [3, 4] and img.m == 1, "Only RGB images are supported in GUI.set_image when fast_gui=True"
            assert img.dtype in [
                ti.f32,
                ti.f64,
                ti.u8,
            ], "Only f32, f64, u8 are supported in GUI.set_image when fast_gui=True"

            vector_to_fast_image(img, self.img)
            return

        if isinstance(img, ScalarField):
            if _ti_core.is_integral(img.dtype) or len(img.shape) != 2:
                # Images of uint is not optimized by xxx_to_image
                self.img = self.cook_image(img.to_numpy())
            else:
                # Type matched! We can use an optimized copy kernel.
                assert img.shape == self.res, "Image resolution does not match GUI resolution"
                tensor_to_image(img, self.img)
                ti.sync()

        elif isinstance(img, taichi.lang.matrix.MatrixField):
            if _ti_core.is_integral(img.dtype):
                self.img = self.cook_image(img.to_numpy())
            else:
                # Type matched! We can use an optimized copy kernel.
                assert img.shape == self.res, "Image resolution does not match GUI resolution"
                assert (
                    img.n in [2, 3, 4] and img.m == 1
                ), "Only greyscale, RG, RGB or RGBA images are supported in GUI.set_image"

                vector_to_image(img, self.img)
                ti.sync()

        elif isinstance(img, np.ndarray):
            self.img = self.cook_image(img)

        else:
            raise ValueError(f"GUI.set_image only takes a Taichi field or NumPy array, not {type(img)}")

        self.core.set_img(self.img.ctypes.data)

    def contour(self, scalar_field, normalize=False):
        """Plot a contour view of a scalar field.

        The input scalar_field will be converted to a Numpy array first, and then plotted
        by the Matplotlib colormap 'Plasma'. Notice this method will automatically perform
        a bilinear interpolation on the field if the size of the field does not match with
        the GUI window size.

        Args:
            scalar_field (ti.field): The scalar field being plotted.
            normalize (bool, Optional): Display the normalized scalar field if set to True.
            Default is False.
        """
        try:
            from matplotlib import cm  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise RuntimeError(
                "Failed to import Matplotlib. Please install it via /\
            `pip install matplotlib` first. "
            )
        scalar_field_np = scalar_field.to_numpy()
        if self.res != scalar_field_np.shape:
            x, y = np.meshgrid(np.linspace(0, 1, self.res[1]), np.linspace(0, 1, self.res[0]))
            x_idx = x * (scalar_field_np.shape[1] - 1)
            y_idx = y * (scalar_field_np.shape[0] - 1)
            x1 = x_idx.astype(int)
            x2 = np.minimum(x1 + 1, scalar_field_np.shape[1] - 1)
            y1 = y_idx.astype(int)
            y2 = np.minimum(y1 + 1, scalar_field_np.shape[0] - 1)
            array_y1 = scalar_field_np[y1, x1] * (1 - (x_idx - x1)) * (1 - (y_idx - y1)) + scalar_field_np[y1, x2] * (
                x_idx - x1
            ) * (1 - (y_idx - y1))
            array_y2 = scalar_field_np[y2, x1] * (1 - (x_idx - x1)) * (y_idx - y1) + scalar_field_np[y2, x2] * (
                x_idx - x1
            ) * (y_idx - y1)
            output = array_y1 + array_y2
        else:
            output = scalar_field_np
        if normalize:
            scalar_max, scalar_min = np.max(output), np.min(output)
            output = (output - scalar_min) / (scalar_max - scalar_min)
        self.set_image(cm.plasma(output))

    def circle(self, pos, color=0xFFFFFF, radius=1):
        """Draws a circle on canvas.

        Args:
            pos (Union[List[int], numpy.array]): The position of the circle.
            color (int, Optional): The color of the circle. Default is 0xFFFFFF.
            radius (Number, Optional): The radius of the circle in pixel. Default is 1.
        """
        self.canvas.circle_single(pos[0], pos[1], color, radius)

    def circles(self, pos, radius=1, color=0xFFFFFF, palette=None, palette_indices=None):
        """Draws a list of circles on canvas.

        Args:
            pos (numpy.array): The positions of the circles.
            radius (Union[Number, numpy.array], optional): The radius of the circles in pixel. \
                Can be either a number, which will be applied to all circles, or a 1D NumPy array of the same length as `pos`. \
                The default is 1.
            color (int, optional): The color of the circles. Default is 0xFFFFFF.
            palette (list[int], optional): The List of colors from which to
                choose to draw. Default is None.
            palette_indices (Union[list[int], ti.field, numpy.array], optional):
                The List of indices that choose color from palette for each
                circle. Shape must match pos. Default is None.

        """
        n = pos.shape[0]
        if len(pos.shape) == 3:
            assert pos.shape[2] == 1
            pos = pos[:, :, 0]

        assert pos.shape == (n, 2)
        pos = np.ascontiguousarray(pos.astype(np.float32))
        # Note: do not use "pos = int(pos.ctypes.data)" here
        # Otherwise pos will get garbage collected by Python
        # and the pointer to its data becomes invalid
        pos_ptr = int(pos.ctypes.data)

        if isinstance(color, np.ndarray):
            assert color.shape == (n,)
            color = np.ascontiguousarray(color.astype(np.uint32))
            color_array = int(color.ctypes.data)
            color_single = 0
        elif isinstance(color, int):
            color_array = 0
            color_single = color
        else:
            raise ValueError("Color must be an ndarray or int (e.g., 0x956333)")

        if palette is not None:
            assert palette_indices is not None, "palette must be used together with palette_indices"

            if isinstance(palette_indices, Field):
                ind_int = palette_indices.to_numpy().astype(np.uint32)
            elif isinstance(palette_indices, list) or isinstance(palette_indices, np.ndarray):
                ind_int = np.array(palette_indices).astype(np.uint32)
            else:
                try:
                    ind_int = np.array(palette_indices)
                except:
                    raise TypeError("palette_indices must be a type that can be converted to numpy.ndarray")

            assert issubclass(ind_int.dtype.type, np.integer), "palette_indices must be an integer array"
            assert ind_int.shape == (n,), "palette_indices must be in 1-d shape with shape (num_particles, )"
            assert min(ind_int) >= 0, "the min of palette_indices must not be less than zero"
            assert max(ind_int) < len(palette), "the max of palette_indices must not exceed the length of palette"
            color_array = np.array(palette, dtype=np.uint32)[ind_int]
            color_array = np.ascontiguousarray(color_array)
            color_array = color_array.ctypes.data

        if isinstance(radius, np.ndarray):
            assert radius.shape == (n,)
            radius = np.ascontiguousarray(radius.astype(np.float32))
            radius_array = int(radius.ctypes.data)
            radius_single = 0
        elif isinstance(radius, numbers.Number):
            radius_array = 0
            radius_single = radius
        else:
            raise ValueError("Radius must be an ndarray or float (e.g., 0.4)")

        self.canvas.circles_batched(n, pos_ptr, color_single, color_array, radius_single, radius_array)

    def triangles(self, a, b, c, color=0xFFFFFF):
        """Draws a list of triangles on canvas.

        Args:
            a (numpy.array): The positions of the first points of triangles.
            b (numpy.array): The positions of the second points of triangles.
            c (numpy.array): The positions of the third points of triangles.
            color (Union[int, numpy.array], optional): The color or colors of triangles.
                Can be either a single color or a list of colors whose shape matches
                the shape of a & b & c. Default is 0xFFFFFF.

        """
        assert a.shape == b.shape
        assert a.shape == c.shape
        n = a.shape[0]
        if len(a.shape) == 3:
            assert a.shape[2] == 1
            a = a[:, :, 0]
            b = b[:, :, 0]
            c = c[:, :, 0]

        assert a.shape == (n, 2)
        a = np.ascontiguousarray(a.astype(np.float32))
        b = np.ascontiguousarray(b.astype(np.float32))
        c = np.ascontiguousarray(c.astype(np.float32))
        # Note: do not use "a = int(a.ctypes.data)" here
        # Otherwise a will get garbage collected by Python
        # and the pointer to its data becomes invalid
        a_ptr = int(a.ctypes.data)
        b_ptr = int(b.ctypes.data)
        c_ptr = int(c.ctypes.data)

        if isinstance(color, np.ndarray):
            assert color.shape == (n,)
            color = np.ascontiguousarray(color.astype(np.uint32))
            color_array = int(color.ctypes.data)
            color_single = 0
        elif isinstance(color, int):
            color_array = 0
            color_single = color
        else:
            raise ValueError('"color" must be an ndarray or int (e.g., 0x956333)')

        self.canvas.triangles_batched(n, a_ptr, b_ptr, c_ptr, color_single, color_array)

    def triangle(self, a, b, c, color=0xFFFFFF):
        """Draws a single triangle on canvas.

        Args:
            a (List[Number]): The position of the first point of triangle. Shape must be 2.
            b (List[Number]): The position of the second point of triangle. Shape must be 2.
            c (List[Number]): The position of the third point of triangle. Shape must be 2.
            color (int, optional): The color of the triangle. Default is 0xFFFFFF.
        """
        self.canvas.triangle_single(a[0], a[1], b[0], b[1], c[0], c[1], color)

    def lines(self, begin, end, radius=1, color=0xFFFFFF):
        """Draw a list of lines on canvas.

        Args:
            begin (numpy.array): The positions of one end of lines.
            end (numpy.array): The positions of the other end of lines.
            radius (Union[Number, numpy.array], optional): The width of lines.
                Can be either a single width or a list of width whose shape matches
                the shape of begin & end. Default is 1.
            color (Union[int, numpy.array], optional): The color or colors of lines.
                Can be either a single color or a list of colors whose shape matches
                the shape of begin & end. Default is 0xFFFFFF.

        """
        assert begin.shape == end.shape
        n = begin.shape[0]
        if len(begin.shape) == 3:
            assert begin.shape[2] == 1
            begin = begin[:, :, 0]
            end = end[:, :, 0]

        assert begin.shape == (n, 2)
        begin = np.ascontiguousarray(begin.astype(np.float32))
        end = np.ascontiguousarray(end.astype(np.float32))
        # Note: do not use "begin = int(begin.ctypes.data)" here
        # Otherwise begin will get garbage collected by Python
        # and the pointer to its data becomes invalid
        begin_ptr = int(begin.ctypes.data)
        end_ptr = int(end.ctypes.data)

        if isinstance(color, np.ndarray):
            assert color.shape == (n,)
            color = np.ascontiguousarray(color.astype(np.uint32))
            color_array = int(color.ctypes.data)
            color_single = 0
        elif isinstance(color, int):
            color_array = 0
            color_single = color
        else:
            raise ValueError("Color must be an ndarray or int (e.g., 0x956333)")

        if isinstance(radius, np.ndarray):
            assert radius.shape == (n,)
            radius = np.ascontiguousarray(radius.astype(np.float32))
            radius_array = int(radius.ctypes.data)
            radius_single = 0
        elif isinstance(radius, numbers.Number):
            radius_array = 0
            radius_single = radius
        else:
            raise ValueError("Radius must be an ndarray or float (e.g., 0.4)")

        self.canvas.paths_batched(
            n,
            begin_ptr,
            end_ptr,
            color_single,
            color_array,
            radius_single,
            radius_array,
        )

    def line(self, begin, end, radius=1, color=0xFFFFFF):
        """Draws a single line on canvas.

        Args:
            begin (List[Number]): The position of one end of line. Shape must be 2.
            end (List[Number]): The position of the other end of line. Shape must be 2.
            radius (Number, optional): The width of line. Default is 1.
            color (int, optional): The color of line. Default is 0xFFFFFF.

        """
        self.canvas.path_single(begin[0], begin[1], end[0], end[1], color, radius)

    @staticmethod
    def _arrow_to_lines(orig, major, tip_scale=0.2, angle=45):
        angle = math.radians(180 - angle)
        c, s = math.cos(angle), math.sin(angle)
        minor1 = np.array([major[:, 0] * c - major[:, 1] * s, major[:, 0] * s + major[:, 1] * c]).swapaxes(0, 1)
        minor2 = np.array([major[:, 0] * c + major[:, 1] * s, -major[:, 0] * s + major[:, 1] * c]).swapaxes(0, 1)
        end = orig + major
        return [
            (orig, end),
            (end, end + minor1 * tip_scale),
            (end, end + minor2 * tip_scale),
        ]

    def arrows(self, orig, direction, radius=1, color=0xFFFFFF, **kwargs):
        """Draw a list arrows on canvas.

        Args:
            orig (numpy.array): The positions where arrows start.
            direction (numpy.array): The directions where arrows point to.
            radius (Union[Number, np.array], optional): The width of arrows. Default is 1.
            color (Union[int, np.array], optional): The color or colors of arrows. Default is 0xffffff.

        """
        for begin, end in self._arrow_to_lines(orig, direction, **kwargs):
            self.lines(begin, end, radius, color)

    def arrow(self, orig, direction, radius=1, color=0xFFFFFF, **kwargs):
        """Draws a single arrow on canvas.

        Args:
            orig (List[Number]): The position where arrow starts. Shape must be 2.
            direction (List[Number]): The direction where arrow points to. Shape must be 2.
            radius (Number, optional): The width of arrow. Default is 1.
            color (int, optional): The color of arrow. Default is 0xFFFFFF.

        """
        orig = np.array([orig])
        direction = np.array([direction])
        for begin, end in self._arrow_to_lines(orig, direction, **kwargs):
            self.line(begin[0], end[0], radius, color)

    def rect(self, topleft, bottomright, radius=1, color=0xFFFFFF):
        """Draws a single rectangle on canvas.

        Args:
            topleft (List[Number]): The position of the topleft corner of rectangle.
                Shape must be 2.
            bottomright (List[Number]): The position of the bottomright corner
                of rectangle. Shape must be 2.
            radius (Number, optional): The width of rectangle's sides. Default is 1.
            color (int, optional): The color of rectangle. Default is 0xFFFFFF.

        """
        a = topleft[0], topleft[1]
        b = bottomright[0], topleft[1]
        c = bottomright[0], bottomright[1]
        d = topleft[0], bottomright[1]
        self.line(a, b, radius, color)
        self.line(b, c, radius, color)
        self.line(c, d, radius, color)
        self.line(d, a, radius, color)

    def text(self, content, pos, font_size=15, color=0xFFFFFF):
        """Draws texts on canvas.

        Args:
            content (str): The text to be drawn on canvas.
            pos (List[Number]): The position where the text is to be put.
            font_size (Number, optional): The font size of the text.
            color (int, optional): The color of the text. Default is 0xFFFFFF.

        """

        # TODO: refactor Canvas::text
        font_size = float(font_size)
        pos = core_vec(*pos)
        r, g, b = hex_to_rgb(color)
        color = core_vec(r, g, b, 1)
        self.canvas.text(content, pos, font_size, color)

    @staticmethod
    def _make_field_base(w, h, bound):
        x = np.linspace(bound / w, 1 - bound / w, w)
        y = np.linspace(bound / h, 1 - bound / h, h)
        base = np.array(np.meshgrid(x, y))
        base = base.swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)
        return base.reshape(w * h, 2)

    def point_field(self, radius, color=0xFFFFFF, bound=0.5):
        """Draws a field of points on canvas.

        Args:
            radius (np.array): The pattern and radius of the field of points.
            color (Union[int, np.array], optional): The color or colors of points.
                Default is 0xFFFFFF.
            bound (Number, optional): The boundary of the field. Default is 0.5.

        """
        assert len(radius.shape) == 2
        base = self._make_field_base(radius.shape[0], radius.shape[1], bound)
        radius = radius.reshape(radius.shape[0] * radius.shape[1])
        self.circles(base, radius=radius, color=color)

    def arrow_field(self, direction, radius=1, color=0xFFFFFF, bound=0.5, **kwargs):
        """Draw a field of arrows on canvas.

        Args:
            direction (np.array): The pattern and direction of the field of arrows.
            color (Union[int, np.array], optional): The color or colors of arrows.
                Default is 0xFFFFFF.
            bound (Number, optional): The boundary of the field. Default is 0.5.

        """
        assert len(direction.shape) == 3
        assert direction.shape[2] == 2
        base = self._make_field_base(direction.shape[0], direction.shape[1], bound)
        direction = direction.reshape(direction.shape[0] * direction.shape[1], 2)
        self.arrows(base, direction, radius=radius, color=color, **kwargs)

    def vector_field(self, vector_field, arrow_spacing=5, color=0xFFFFFF):
        """Display a vector field on canvas.

        Args:
            vector_field (ti.Vector.field): The vector field being displayed.
            arrow_spacing (int, optional): The spacing between vectors.
            color (Union[int, np.array], optional): The color of vectors.

        """
        v_np = vector_field.to_numpy()
        v_norm = np.linalg.norm(v_np, axis=-1)
        nx, ny, ndim = v_np.shape
        max_magnitude = np.max(v_norm)  # Find the largest vector magnitude

        # The largest vector should occupy 10% of the window
        scale_factor = 0.1 / (max_magnitude + 1e-16)

        x = np.arange(0, 1, arrow_spacing / nx)
        y = np.arange(0, 1, arrow_spacing / ny)
        X, Y = np.meshgrid(x, y)
        begin = np.dstack((X, Y)).reshape(-1, 2, order="F")
        incre = (v_np[::arrow_spacing, ::arrow_spacing] * scale_factor).reshape(-1, 2, order="C")
        self.arrows(orig=begin, direction=incre, radius=1, color=color)

    def show(self, file=None):
        """Shows the frame content in the gui window, or save the content to an
        image file.

        Args:
            file (str, optional): output filename. The default is `None`, and
                the frame content is displayed in the gui window. If it's a valid
                image filename the frame will be saved as the specified image.
        """
        self.core.update()
        if file:
            self.core.screenshot(file)
        self.frame += 1
        self.clear()

    # Event system

    class EventFilter:
        """A set to store detected user events."""

        def __init__(self, *e_filter):
            self.filter = set()
            for ent in e_filter:
                if isinstance(ent, (list, tuple)):
                    e_type, key = ent
                    ent = (e_type, key)
                self.filter.add(ent)

        def match(self, e):
            """Check if a specified event `e` is among the detected events."""
            if (e.type, e.key) in self.filter:
                return True
            if e.type in self.filter:
                return True
            if e.key in self.filter:
                return True
            return False

    def has_key_event(self):
        """Check if there is any key event registered.

        Returns:
            bool: whether or not there is any key event registered.
        """
        return self.core.has_key_event()

    def get_event(self, *e_filter):
        """Checks if the specified event is triggered.

        Args:
            *e_filter (ti.GUI.EVENT): The specific event to be checked.

        Returns:
            bool: whether or not the specified event is triggered.
        """
        for e in self.get_events(*e_filter):
            self.event = e
            return True
        else:
            return False

    def get_events(self, *e_filter):
        """Gets a list of events that are triggered.

        Args:
            *e_filter (List[ti.GUI.EVENT]): The type of events to be filtered.

        Returns:
            :class:`~taichi.misc.gui.GUI.EVENT`: A list of events that are triggered.
        """
        e_filter = e_filter and GUI.EventFilter(*e_filter) or None

        while True:
            if not self.has_key_event():
                break
            e = self.get_key_event()
            if e_filter is None or e_filter.match(e):  # pylint: disable=E1101
                yield e

    def get_key_event(self):
        """Gets keyboard triggered event.

        Returns:
            :class:`~taichi.misc.gui.GUI.EVENT`: The keyboard triggered event.
        """
        self.core.wait_key_event()

        e = GUI.Event()
        event = self.core.get_key_event_head()

        e.type = event.type
        e.key = event.key
        e.pos = self.core.canvas_untransform(event.pos)
        e.pos = (e.pos[0], e.pos[1])
        e.modifier = []

        if e.key == GUI.WHEEL:
            e.delta = event.delta
        else:
            e.delta = (0, 0)

        for mod in ["Shift", "Alt", "Control"]:
            if self.is_pressed(mod):
                e.modifier.append(mod)

        if e.type == GUI.PRESS:
            self.key_pressed.add(e.key)
        else:
            self.key_pressed.discard(e.key)

        self.core.pop_key_event_head()
        return e

    def is_pressed(self, *keys):
        """Checks if any key among a set of specified keys is pressed.

        Args:
            *keys (Union[str, List[str]]): The keys to be listened to.

        Returns:
            bool: whether or not any key among the specified keys is pressed.
        """
        for key in keys:
            if key in ["Shift", "Alt", "Control"]:
                if key + "_L" in self.key_pressed or key + "_R" in self.key_pressed:
                    return True
            if key in self.key_pressed:
                return True
        else:
            return False

    def get_cursor_pos(self):
        """Returns the current position of mouse as a pair of floats
        in the range `[0, 1] x [0, 1]`.

        The origin of the coordinates system is located at the lower left
        corner, with `+x` direction points to the right, and `+y` direcntion
        points upward.

        Returns:
            The current position of mouse.
        """
        pos = self.core.get_cursor_pos()
        return pos[0], pos[1]

    @property
    def running(self):
        """Returns whether this gui is running or not.

        Returns:
            bool: whether this gui is running or not.
        """
        return not self.core.should_close

    @running.setter
    def running(self, value):
        """Sets the running status of this gui. `True` for running
        and `False` for stop.

        Args:
            value (bool): `True/False` for running/stop.
        """
        if value:
            self.core.should_close = 0
        elif not self.core.should_close:
            self.core.should_close = 1

    @property
    def fps_limit(self):
        """Gets the maximum fps of this gui.

        Returns:
            int: the maximum fps.
        """
        if self.core.frame_delta_limit == 0:
            return None
        return 1 / self.core.frame_delta_limit

    @fps_limit.setter
    def fps_limit(self, value):
        """Set the maximum fps for the gui. This is the maximum number of
        frames that the gui can show in one second.

        Args:
            value (int): maximum fps.
        """
        if value is None:
            self.core.frame_delta_limit = 0
        else:
            self.core.frame_delta_limit = 1 / value


def rgb_to_hex(c):
    """Converts rgb color format to hex color format.

    Args:
        c (List[int]): The rgb representation of color.

    Returns:
        The hex representation of color.
    """

    def to255(x):
        return np.clip(np.int32(x * 255), 0, 255)

    return (to255(c[0]) << 16) + (to255(c[1]) << 8) + to255(c[2])


def hex_to_rgb(color):
    """Converts hex color format to rgb color format.

    Args:
        color (int): The hex representation of color.

    Returns:
        The rgb representation of color.
    """
    r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return r / 255, g / 255, b / 255


def core_veci(*args):
    if isinstance(args[0], _ti_core.Vector2i):
        return args[0]
    if isinstance(args[0], _ti_core.Vector3i):
        return args[0]
    if isinstance(args[0], tuple):
        args = tuple(*args)
    if len(args) == 2:
        return _ti_core.Vector2i(int(args[0]), int(args[1]))
    if len(args) == 3:
        return _ti_core.Vector3i(int(args[0]), int(args[1]), int(args[2]))
    if len(args) == 4:
        return _ti_core.Vector4i(int(args[0]), int(args[1]), int(args[2]), int(args[3]))
    assert False, type(args[0])


def core_vec(*args):
    if isinstance(args[0], _ti_core.Vector2f):
        return args[0]
    if isinstance(args[0], _ti_core.Vector3f):
        return args[0]
    if isinstance(args[0], _ti_core.Vector4f):
        return args[0]
    if isinstance(args[0], _ti_core.Vector2d):
        return args[0]
    if isinstance(args[0], _ti_core.Vector3d):
        return args[0]
    if isinstance(args[0], _ti_core.Vector4d):
        return args[0]
    if isinstance(args[0], tuple):
        args = tuple(*args)
    if _ti_core.get_default_float_size() == 4:
        if len(args) == 2:
            return _ti_core.Vector2f(float(args[0]), float(args[1]))
        if len(args) == 3:
            return _ti_core.Vector3f(float(args[0]), float(args[1]), float(args[2]))
        if len(args) == 4:
            return _ti_core.Vector4f(float(args[0]), float(args[1]), float(args[2]), float(args[3]))
        assert False, type(args[0])
    else:
        if len(args) == 2:
            return _ti_core.Vector2d(float(args[0]), float(args[1]))
        if len(args) == 3:
            return _ti_core.Vector3d(float(args[0]), float(args[1]), float(args[2]))
        if len(args) == 4:
            return _ti_core.Vector4d(float(args[0]), float(args[1]), float(args[2]), float(args[3]))
        assert False, type(args[0])


__all__ = [
    "GUI",
    "rgb_to_hex",
    "hex_to_rgb",
]
