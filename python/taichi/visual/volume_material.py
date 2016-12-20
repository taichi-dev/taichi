from taichi.util import *
from taichi.core import tc_core
from taichi.visual.asset_manager import asset_ptr_to_id

class VolumeMaterial:
    def __init__(self, name, **kwargs):
        kwargs = asset_ptr_to_id(kwargs)
        self.c = tc_core.create_volume_material(name)
        self.c.initialize(config_from_dict(kwargs))
