import math

from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture


def create_scene():
    downsample = 2
    width, height = 960 / downsample, 540 / downsample
    camera = Camera('pinhole', width=width, height=height, fov=120,
                    origin=(0, 0, 10), look_at=(0, -0.3, 0), up=(0, 1, 0))

    scene = Scene()
    with scene:
        scene.set_camera(camera)
        rep = Texture.create_taichi_wallpaper(20, rotation=0, scale=0.95)
        material = SurfaceMaterial('pbr', diffuse_map=rep.id)
        #scene.add_mesh(Mesh('holder', material=material, translate=(0, -1, -7), scale=2))

        envmap_texture = Texture.create_taichi_wallpaper(8)
        envmap_texture.show()
        envmap = EnvironmentMap('base', texture=envmap_texture.id)
        scene.set_environment_map(envmap)
    return scene


if __name__ == '__main__':
    renderer = Renderer(overwrite=True)
    renderer.initialize(preset='pt', scene=create_scene())
    renderer.set_post_processor(LDRDisplay(exposure=4, bloom_radius=0.00))
    renderer.render(800)
