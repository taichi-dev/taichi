from taichi._lib import core as _ti_core

from .camera import Camera  # pylint: disable=unused-import
from .canvas import Canvas  # pylint: disable=unused-import
from .constants import *  # pylint: disable=unused-import,wildcard-import
from .imgui import Gui  # pylint: disable=unused-import
from .scene import Scene  # pylint: disable=unused-import
from .window import Window  # pylint: disable=unused-import
from .utils import check_ggui_availability


def make_camera():
    check_ggui_availability()
    return Camera(_ti_core.PyCamera())


ProjectionMode = _ti_core.ProjectionMode if _ti_core.GGUI_AVAILABLE else None
