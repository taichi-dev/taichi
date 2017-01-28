from math import *

import taichi as tc
from taichi.misc.util import *


def create_scene():
    downsample = 2
    width, height = 800 / downsample, 800 / downsample
    camera = tc.Camera('pinhole', width=width, height=height, fov=60,
                       origin=(0, 3, 10), look_at=(0, 0, 0), up=(0, 1, 0))

    scene = tc.Scene()

    with scene:
        scene.set_camera(camera)

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
                       translate=(3, 3, 0), scale=0.4, rotation=(0, 0, 180))
        scene.add_mesh(mesh)

    return scene

if __name__ == '__main__':
    renderer = tc.Renderer(output_dir='sdf', overwrite=True)
    renderer.initialize(preset='pt_sdf', scene=create_scene(), max_path_length=30)
    renderer.set_post_processor(tc.post_process.LDRDisplay(bloom_radius=0.0))
    renderer.render(10000, 20)
