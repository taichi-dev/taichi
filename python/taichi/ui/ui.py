from taichi._lib import core as _ti_core

from .camera import Camera
from .canvas import Canvas
from .constants import *
from .imgui import Gui
from .scene import Scene
from .window import Window
from .utils import check_ggui_availability


def make_camera():
    check_ggui_availability()
    return Camera(_ti_core.PyCamera())


ProjectionMode = _ti_core.ProjectionMode if _ti_core.GGUI_AVAILABLE else None
