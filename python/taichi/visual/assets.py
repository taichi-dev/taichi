import taichi as tc

from texture import Texture
from surface_material import SurfaceMaterial

class Materials:
    def __init__(self):
        self.materials = {}

    def get_material(self, name):
        '''
        if name not in self.materials:
            self.materials[name] = getattr(self, 'get_material_' + name)()
        return self.materials[name]
        '''
        # Let just waste some memory for easier implementation...
        mat = getattr(self, 'get_material_' + name)()
        return mat

    def get_material_mirror(self):
        material = tc.create_surface_material('pbr')
        material.initialize(P(diffuse=(0.0, 0.0, 0.0), specular=(1.0, 1.0, 1.0), glossiness=-1, transparent=False))
        return material

    def get_material_gold(self):
        return SurfaceMaterial('pbr', diffuse=(0.0, 0.0, 0.0), specular=(1.0, 0.9, 0.6), glossiness=-1,
                               transparent=False)

    def get_material_glossy(self):
        return SurfaceMaterial('pbr', diffuse=(0.0, 0.0, 0.0), specular=(0.5, 0.5, 0.3), glossiness=300,
                                   transparent=False)

    def get_material_wall(self):
        material = tc.create_surface_material('diffuse')
        rep = Texture.create_taichi_wallpaper(20)
        material.initialize(P(diffuse_map=rep.id))
        return material

    def get_material_diffuse_white(self):
        return SurfaceMaterial('diffuse', diffuse=(1, 1, 1))

    def get_material_glass(self):
        material = tc.create_surface_material('pbr')
        material.initialize(P(diffuse=(0, 0, 0), specular=(0.0, 0.0, 0.0), glossiness=-1,
                              transparent=True, ior=1.5))
        return material

    def get_material_dark_grey(self):
        material = tc.create_surface_material('pbr')
        material.initialize(P(diffuse=(0.3, 0.3, 0.3), specular=(0.0, 0.0, 0.0), glossiness=-1,
                              transparent=False))
        return material

    def get_material_interface(self):
        material = tc.create_surface_material('plain_interface')
        material.initialize(P())
        vol = tc.create_volume_material("homogeneous")
        vol.initialize(P(scattering=1, absorption=0))
        material.set_internal_material(vol)
        return material

    def get_material_snow(self):
        material = tc.create_surface_material('plain_interface')
        material.initialize(P())
        vol = tc.create_volume_material("homogeneous")
        vol.initialize(P(scattering=30, absorption=0))
        material.set_internal_material(vol)
        return material

    def get_material_snow_nosss(self):
        material = tc.create_surface_material('diffuse')
        material.initialize(P(diffuse=(1, 1, 1)))
        return material


materials = Materials()
