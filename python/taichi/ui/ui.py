from taichi._lib import core as _ti_core

GGUI_AVAILABLE = _ti_core.GGUI_AVAILABLE

if GGUI_AVAILABLE:

    from .camera import Camera  # pylint: disable=unused-import
    from .canvas import Canvas  # pylint: disable=unused-import
    from .constants import *  # pylint: disable=unused-import,wildcard-import
    from .imgui import Gui  # pylint: disable=unused-import
    from .scene import Scene  # pylint: disable=unused-import
    from .window import Window  # pylint: disable=unused-import

    def make_camera():
        return Camera(_ti_core.PyCamera())

    ProjectionMode = _ti_core.ProjectionMode

else:

    def err_no_ggui():
        raise Exception("GGUI Not Available")

    class Window:
        def __init__(self, name, res, vsync=False):
            err_no_ggui()

    class Scene:
        def __init__(self):
            err_no_ggui()

    def make_camera():
        err_no_ggui()
