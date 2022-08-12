import warnings

from taichi._lib import core as _ti_core

from .camera import Camera
from .canvas import Canvas  # pylint: disable=unused-import
from .constants import *  # pylint: disable=unused-import,wildcard-import
from .imgui import Gui  # pylint: disable=unused-import
from .scene import Scene  # pylint: disable=unused-import
from .utils import check_ggui_availability  # pylint: disable=unused-import
from .window import Window  # pylint: disable=unused-import


def make_camera():
    """Return an instance of :class:`~taichi.ui.Camera`. This is an deprecated
    interface, please construct `~taichi.ui.Camera` directly.

    Example::

        >>> camera = ti.ui.make_camera()
    """
    warnings.warn(
        "`ti.ui.make_camera()` is deprecated, please use `ti.ui.Camera()` instead",
        DeprecationWarning)
    return Camera()


# ----------------------
ProjectionMode = _ti_core.ProjectionMode if _ti_core.GGUI_AVAILABLE else None
"""Camera projection mode, 0 for perspective and 1 for orthogonal.
"""
