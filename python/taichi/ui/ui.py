from taichi.lib.core import ti_core as _ti_core

if _ti_core.GGUI_AVAILABLE:

    from .camera import Camera  # pylint: disable=unused-import


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
