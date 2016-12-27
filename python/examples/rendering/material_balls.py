import taichi as tc
from math import *
import random
import colorsys
from taichi.util import *


def create_scene():
    downsample = 1
    width, height = 960 / downsample, 540 / downsample
    camera = tc.Camera('thinlens', width=width, height=height, fov=40,
                       origin=(0, 20, 40), look_at=(0, 0, 0), up=(0, 1, 0), aperture=0.3)

    scene = tc.Scene()

    base_mesh = tc.geometry.create_sphere((50, 50), smooth=True)

    materials = [
        tc.SurfaceMaterial('diffuse', color=(1, 0, 0)),
        tc.SurfaceMaterial('diffuse', color=(0, 1, 0)),
        tc.SurfaceMaterial('diffuse', color=(0, 0, 1)),
        tc.SurfaceMaterial('reflective', color=(1, 1, 1)),
        tc.SurfaceMaterial('glossy', color=(1, 1, 1), glossiness=(10, 10, 10)),
        tc.SurfaceMaterial('refractive', color=(1, 1, 1), ior=2.5),
        tc.SurfaceMaterial('pbr', diffuse=(1, 0, 0), specular=(0, 1, 0), glossiness=(100, 0, 0)),
    ]

    with scene:
        scene.set_camera(camera)

        for i, mat in enumerate(materials):
            scene.add_mesh(tc.Mesh(base_mesh, mat, translate=(i * 3 - 10, 1.3, 0), scale=1))

        # Ground
        tex = (((tc.Texture('perlin') + 1) * 6).zoom((0.6, 0.6, 0.6))).fract() * (1.0, 0.7, 0.4)
        scene.add_mesh(tc.Mesh('plane', tc.SurfaceMaterial('pbr', diffuse_map=tex, specular=(0.05, 0.03, 0.02),
                                                           glossiness=100, glossy=(0.01, 0.01, 0.01)), scale=200,
                               translate=(0, 0, 0), rotation=(0, 0, 0)))

        envmap = tc.EnvironmentMap('base', filepath='../../taichi_assets/envmaps/schoenbrunn-front_hd.hdr')
        envmap.set_transform(tc.core.Matrix4(1.0).rotate_euler(Vector(0, 30, 0)))
        scene.set_environment_map(envmap)

    return scene


if __name__ == '__main__':
    renderer = tc.Renderer('../output/frames/geometry.png', overwrite=True)
    renderer.initialize(preset='pt', scene=create_scene(), luminance_clamping=1000)
    renderer.set_post_processor(tc.post_process.LDRDisplay(exposure=1, bloom_radius=0.0))
    renderer.render(10000, 20)
