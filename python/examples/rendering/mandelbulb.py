from math import *
import shutil

import taichi as tc
import gc
from taichi.misc.util import *


def create_scene():
    downsample = 2
    width, height = 800 / downsample, 800 / downsample
    camera = tc.Camera('thinlens', width=width, height=height, fov=90, aperture=0.0, focus=(0, 0, 0),
                       origin=(3, 0, 0), look_at=(0, 0, 0), up=(0, 1, 0))

    scene = tc.Scene()

    with scene:
        scene.set_camera(camera)

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
                       translate=(0, 10, 0), scale=2, rotation=(0, 0, 180))
        scene.add_mesh(mesh)

    return scene


if __name__ == '__main__':
    uw = tc.UnitWatcher(tc.settings.get_asset_path('units/sdf/mandelbulb.cpp'))

    while True:
        if uw.need_update():
            renderer = None
            sdf = None
            gc.collect()
            uw.update()
            sdf = tc.core.create_sdf('mandelbulb_sdf')
            sdf_id = tc.core.register_sdf(sdf)
            renderer = tc.Renderer(output_dir='sdf', overwrite=True)
            renderer.initialize(preset='pt_sdf', max_path_length=4, scene=create_scene(), sdf=sdf_id, num_threads=1)
            renderer.set_post_processor(tc.post_process.LDRDisplay(bloom_radius=0.0))

        renderer.render(1)
