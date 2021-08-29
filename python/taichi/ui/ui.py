import pathlib

from taichi.core import ti_core as _ti_core
from taichi.lang.impl import default_cfg, field
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel
from taichi.lang.ops import get_addr

if _ti_core.GGUI_AVAILABLE:

    from .camera import Camera
    from .canvas import Canvas
    from .constants import *
    from .gui import Gui
    from .scene import Scene
    from .window import Window

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
