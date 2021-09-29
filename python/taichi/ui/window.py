import pathlib

from taichi.core import ti_core as _ti_core
from taichi.lang.impl import default_cfg
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel
from taichi.lang.ops import get_addr

from .canvas import Canvas
from .constants import *
from .gui import Gui
from .utils import get_field_info


class Window(_ti_core.PyWindow):
    """The window class.

    Args:
        name (str): name of the window.
        res (Tuple[Int]): resolution (width, height) of the window, in pixels.
        layout (vsync): whether or not vertical sync should be enabled.
    """
    def __init__(self, name, res, vsync=False):
        package_path = str(pathlib.Path(__file__).parent.parent)

        ti_arch = default_cfg().arch
        is_packed = default_cfg().packed
        super().__init__(name, res, vsync, package_path, ti_arch, is_packed)

    @property
    def running(self):
        return self.is_running()

    @running.setter
    def running(self, value):
        self.set_is_running(value)

    def get_events(self, tag=None):
        """ Obtain a list of unprocessed events.

        Args:
            tag (str): A tag used for filtering events. If it is None, then all events are returned.
        """
        if tag == None:
            return super().get_events(_ti_core.EventType.Any)
        elif tag == PRESS:
            return super().get_events(_ti_core.EventType.Press)
        elif tag == RELEASE:
            return super().get_events(_ti_core.EventType.Release)
        raise Exception("unrecognized event tag")

    def get_event(self, tag=None):
        """ Returns whether or not a event that matches tag has occurred.

        If tag is None, then no filters are applied. If this function returns `True`, the `event` property of the window will be set to the corresponding event.

        """
        if tag == None:
            return super().get_event(_ti_core.EventType.Any)
        elif tag == PRESS:
            return super().get_event(_ti_core.EventType.Press)
        elif tag == RELEASE:
            return super().get_event(_ti_core.EventType.Release)
        raise Exception("unrecognized event tag")

    def is_pressed(self, *keys):
        for k in keys:
            if super().is_pressed(k):
                return True
        return False

    def get_canvas(self):
        """Returns a canvas handle. See :class`~taichi.ui.canvas.Canvas` """
        return Canvas(super().get_canvas())

    @property
    def GUI(self):
        """Returns a IMGUI handle. See :class`~taichi.ui.ui.Gui` """
        return Gui(super().GUI())
