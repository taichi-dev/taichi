from taichi._lib import core as _ti_core

from .camera import Camera
from .canvas import Canvas  # pylint: disable=unused-import
from .constants import *  # pylint: disable=unused-import,wildcard-import
from .imgui import Gui  # pylint: disable=unused-import
from .scene import Scene  # pylint: disable=unused-import
from .utils import check_ggui_availability
from .window import Window  # pylint: disable=unused-import


def make_camera():
    """Return an instance of :class:`~taichi.ui.Camera`. This is the
    recommended way to create a camera in ggui.

    You should also mannually set the camera parameters like `camera.position`,
    `camera.lookat`, `camera.up`, etc. The default settings may not work for
    your scene.

    Example::

        >>> camera = ti.ui.make_camera()
    """
    check_ggui_availability()
    return Camera(_ti_core.PyCamera())


ProjectionMode = _ti_core.ProjectionMode if _ti_core.GGUI_AVAILABLE else None
