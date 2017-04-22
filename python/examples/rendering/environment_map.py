import math
import time

from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture


def create_scene():
    downsample = 1
    width, height = 512 / downsample, 512 / downsample
    camera = Camera('pinhole', width=width, height=height, fov=120,
                    origin=(0, -5, 10), look_at=(0, 0, 0), up=(0, 1, 0))

    scene = Scene()
    with scene:
        scene.set_camera(camera)
        tex = Texture.create_taichi_wallpaper(20, rotation=0, scale=0.95) * 0.9
        material = SurfaceMaterial('microfacet', color=(1.0, 1, 0.8), roughness_map=tex.id, f0=1)
        for i in range(-7, 5):
            scene.add_mesh(Mesh('sphere', material=material, translate=(i, -i * 1.6, -math.sin(i * 0.1)), scale=0.7))

        envmap_texture = Texture('sky', height=0.5, direction=0.3)
        envmap_texture.show(res=(500, 500))
        envmap = EnvironmentMap('base', texture=envmap_texture.id, res=(1024, 1024))
        scene.set_environment_map(envmap)
    return scene


if __name__ == '__main__':
    renderer = Renderer(overwrite=True)
    renderer.initialize(preset='pt', scene=create_scene())
    renderer.set_post_processor(FilmicToneMapping(exposure=15, bloom_radius=0.01, gamma=1))
    renderer.render(800)
