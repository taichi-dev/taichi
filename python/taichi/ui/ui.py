from taichi._lib import core as _ti_core

from .camera import Camera  # pylint: disable=unused-import
from .canvas import Canvas  # pylint: disable=unused-import
from .constants import *  # pylint: disable=unused-import,wildcard-import
from .imgui import Gui  # pylint: disable=unused-import
from .scene import Scene  # pylint: disable=unused-import
from .utils import check_ggui_availability  # pylint: disable=unused-import
from .window import Window  # pylint: disable=unused-import


# ----------------------
ProjectionMode = _ti_core.ProjectionMode if _ti_core.GGUI_AVAILABLE else None
"""Camera projection mode, 0 for perspective and 1 for orthogonal.
"""


def make_camera():
    return Camera()
