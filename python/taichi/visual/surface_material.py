from taichi.util import *
from taichi.core import tc_core
import asset_manager

class SurfaceMaterial:
    def __init__(self, name, **kwargs):
        self.c = tc_core.create_surface_material(name)
        kwargs = asset_manager.asset_ptr_to_id(kwargs)
        self.c.initialize(config_from_dict(kwargs))
        self.id = tc_core.register_surface_material(self.c)

    def set_internal_material(self, vol):
        self.c.set_internal_material(vol.c)

