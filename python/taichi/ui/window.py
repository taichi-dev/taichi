import pathlib

from taichi.core import ti_core as _ti_core
from taichi.core.primitive_types import *
from taichi.lang.impl import default_cfg
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel
from taichi.lang.ops import get_addr

from .canvas import Canvas
from .gui import Gui
from .utils import *

SHIFT = 'Shift'
ALT = 'Alt'
CTRL = 'Control'
ESCAPE = 'Escape'
RETURN = 'Return'
TAB = 'Tab'
BACKSPACE = 'BackSpace'
SPACE = 'Space'
UP = 'Up'
DOWN = 'Down'
LEFT = 'Left'
RIGHT = 'Right'
CAPSLOCK = 'CapsLock'
LMB = 'LMB'
MMB = 'MMB'
RMB = 'RMB'

# Event types
PRESS = "Press"
RELEASE = "Release"


class Window(_ti_core.PyWindow):
    def __init__(self, name, res, vsync=False):
        package_path = str(pathlib.Path(__file__).parent.parent)

        ti_arch = default_cfg().arch
        super().__init__(name, res, vsync, package_path, ti_arch)

    @property
    def running(self):
        return self.is_running()

    @running.setter
    def running(self, value):
        self.set_is_running(value)

    def get_events(self, tag=None):
        if tag == None:
            return super().get_events(_ti_core.EventType.Any)
        elif tag == PRESS:
            return super().get_events(_ti_core.EventType.Press)
        elif tag == RELEASE:
            return super().get_events(_ti_core.EventType.Release)
        raise Exception("unrecognized event tag")

    def get_event(self, tag=None):
        if tag == None:
            return super().get_event(_ti_core.EventType.Any)
        elif tag == PRESS:
            return super().get_event(_ti_core.EventType.Press)
        elif tag == RELEASE:
            return super().get_events(_ti_core.EventType.Release)
        raise Exception("unrecognized event tag")

    def is_pressed(self, *keys):
        for k in keys:
            if super().is_pressed(k):
                return True
        return False

    def get_canvas(self):
        return Canvas(super().get_canvas())

    @property
    def GUI(self):
        return Gui(super().GUI())
