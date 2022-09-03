import pathlib
import warnings

import numpy
from taichi._kernels import (arr_vulkan_layout_to_arr_normal_layout,
                             arr_vulkan_layout_to_field_normal_layout)
from taichi._lib import core as _ti_core
from taichi.lang._ndarray import Ndarray
from taichi.lang.impl import Field, default_cfg, get_runtime
from taichi.ui.staging_buffer import get_depth_ndarray

from taichi import f32

from .canvas import Canvas
from .constants import PRESS, RELEASE
from .imgui import Gui
from .utils import check_ggui_availability


class Window:
    """The window class.

    Args:
        name (str): Window title.
        res (tuple[int]): resolution (width, height) of the window, in pixels.
        vsync (bool): whether or not vertical sync should be enabled.
        show_window (bool): where or not display the window after initialization.
    """
    def __init__(self, name, res, vsync=False, show_window=True):
        check_ggui_availability()
        package_path = str(pathlib.Path(__file__).parent.parent)
        ti_arch = default_cfg().arch
        is_packed = default_cfg().packed
        self.window = _ti_core.PyWindow(get_runtime().prog, name, res, vsync,
                                        show_window, package_path, ti_arch,
                                        is_packed)

    @property
    def running(self):
        """Check whether this window is running or not."""
        return self.window.is_running()

    @running.setter
    def running(self, value):
        """Set the running status of this window.

        Example::

            >>> window.running = False
        """
        self.window.set_is_running(value)

    @property
    def event(self):
        """Get the current unprocessed event.
        """
        return self.window.get_current_event()

    @event.setter
    def event(self, value):
        """Set the current unprocessed event.
        """
        self.window.set_current_event(value)

    def get_events(self, tag=None):
        """Get the current list of unprocessed events.

        Args:
            tag (str): A tag used for filtering events. \
                If it is None, then all events are returned.
        """
        if tag is None:
            return self.window.get_events(_ti_core.EventType.Any)
        if tag is PRESS:
            return self.window.get_events(_ti_core.EventType.Press)
        if tag is RELEASE:
            return self.window.get_events(_ti_core.EventType.Release)
        raise Exception("unrecognized event tag")

    def get_event(self, tag=None):
        """Returns whether or not a event that matches tag has occurred.

        If tag is None, then no filters are applied. If this function
        returns `True`, the `event` property of the window will be set
        to the corresponding event.
        """
        if tag is None:
            return self.window.get_event(_ti_core.EventType.Any)
        if tag is PRESS:
            return self.window.get_event(_ti_core.EventType.Press)
        if tag is RELEASE:
            return self.window.get_event(_ti_core.EventType.Release)
        raise Exception("unrecognized event tag")

    def is_pressed(self, *keys):
        """Checks if any of a set of specified keys is pressed.

        Args:
            keys (list[:mod:`~taichi.ui.constants`]): The keys to be matched.

        Returns:
            bool: `True` if any key among `keys` is pressed, else `False`.
        """
        for k in keys:
            if self.window.is_pressed(k):
                return True
        return False

    def get_canvas(self):
        """Returns a canvas handle. See :class`~taichi.ui.canvas.Canvas` """
        return Canvas(self.window.get_canvas())

    @property
    def GUI(self):
        """Returns a IMGUI handle. See :class`~taichi.ui.ui.Gui` This is an
        deprecated interface, please use `~taichi.ui.Window.get_gui` instead.
        """
        return self.get_gui()

    def get_gui(self):
        """Returns a IMGUI handle. See :class`~taichi.ui.ui.Gui`"""
        return Gui(self.window.GUI())

    def get_cursor_pos(self):
        """Get current cursor position, in the range `[0, 1] x [0, 1]`.
        """
        return self.window.get_cursor_pos()

    def show(self):
        """Display this window.
        """
        return self.window.show()

    def get_window_shape(self):
        """Return the shape of window.
        Return:
            tuple : (width, height)
        """
        return self.window.get_window_shape()

    def write_image(self, filename):
        """Save the window content to an image file. This is an deprecated
        interface; please use `save_image` instead.

        Args:
            filename (str): output filename.
        """
        warnings.warn(
            "`Window.write_image()` is renamed to `Window.save_image()`",
            DeprecationWarning)
        return self.save_image(filename)

    def save_image(self, filename):
        """Save the window content to an image file.

        Args:
            filename (str): output filename.
        """
        return self.window.write_image(filename)

    def get_depth_buffer(self, depth):
        """fetch the depth information of current scene to ti.ndarray/ti.field
           (support copy from vulkan to cuda/cpu which is a faster version)
        Args:
            depth(ti.ndarray/ti.field): [window_width, window_height] carries depth information.
        """
        if not (len(depth.shape) == 2 and depth.dtype == f32):
            raise Exception("Only Support 2d-shape and ti.f32 data format.")
        if not isinstance(depth, (Ndarray, Field)):
            raise Exception("Only Support Ndarray and Field data type.")
        tmp_depth = get_depth_ndarray(self.window)
        self.window.copy_depth_buffer_to_ndarray(tmp_depth.arr)
        if isinstance(depth, Ndarray):
            arr_vulkan_layout_to_arr_normal_layout(tmp_depth, depth)
        else:
            arr_vulkan_layout_to_field_normal_layout(tmp_depth, depth)

    def get_depth_buffer_as_numpy(self):
        """Get the depth information of current scene to numpy array.

        Returns:
            2d numpy array: [width, height] with (0.0~1.0) float-format.
        """
        tmp_depth = get_depth_ndarray(self.window)
        self.window.copy_depth_buffer_to_ndarray(tmp_depth.arr)
        depth_numpy_arr = numpy.zeros(self.get_window_shape(),
                                      dtype=numpy.float32)
        arr_vulkan_layout_to_arr_normal_layout(tmp_depth, depth_numpy_arr)
        return depth_numpy_arr

    def get_image_buffer_as_numpy(self):
        """Get the window content to numpy array.

        Returns:
            3d numpy array: [width, height, channels] with (0.0~1.0) float-format color.
        """
        return self.window.get_image_buffer_as_numpy()

    def destroy(self):
        """Destroy this window. The window will be unavailable then.
        """
        return self.window.destroy()
