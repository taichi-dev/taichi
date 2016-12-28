import taichi as tc
from math import *
import random
import colorsys
from taichi.util import *


def create_scene():
    downsample = 2
    width, height = 600 / downsample, 800 / downsample
    camera = tc.Camera('pinhole', width=width, height=height, fov=30,
                       origin=(0, 4, 40), look_at=(0, 4, 0), up=(0, 1, 0))

    scene = tc.Scene()

    radius = lambda x: exp(sin(x * 2 * pi + 0.1) + x + sin(150 * x) * 0.02) / 3
    surf = lambda p: tc.geometry.rotate_y(Vector(radius(p.x), p.x * 4, 0), p.y * 2 * pi)

    tex = ((tc.Texture('perlin') + 1) * 5).zoom(zoom=(10, 10, 2)).fract() * (1.0, 0.4, 0.2)

    with scene:
        scene.set_camera(camera)

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('diffuse', color=(1, 1, 1)),
                       translate=(0, 0, 0), scale=40, rotation=(0, 0, 0))
        scene.add_mesh(mesh)

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('diffuse', color=(1, 1, 1)),
                       translate=(0, 0, -40), scale=40, rotation=(90, 0, 0))
        scene.add_mesh(mesh)

        mesh = tc.geometry.create_mesh_from_functions((150, 150), surf, smooth=False)
        scene.add_mesh(tc.Mesh(mesh, tc.SurfaceMaterial('pbr',
                                                        diffuse_map=tex),
                               translate=(0, 0, 0), scale=2))

        # Lights

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(1, 0.9, 0.6)),
                       translate=(-10, 10, 30), scale=1, rotation=(-110, -45, 0))
        scene.add_mesh(mesh)

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(0.01, 0.02, 0.04)),
                       translate=(10, 10, 30), scale=3, rotation=(-110, 45, 0))
        scene.add_mesh(mesh)

        with tc.transform_scope(translate=(9, 10, -10), scale=(0.3, 1.3, 0.3), rotation=(88, -27, 0)):
            mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(100, 100, 10)))
            scene.add_mesh(mesh)
            surf = lambda p: tc.geometry.rotate_y(Vector(p.x * 3, p.x * p.x * 4 - 2, 0), p.y * 2 * pi)
            bowl = tc.geometry.create_mesh_from_functions((50, 50), surf)
            mesh = tc.Mesh(bowl, tc.SurfaceMaterial('diffuse', color=(0, 0, 0)))
            scene.add_mesh(mesh)

    return scene


if __name__ == '__main__':
    renderer = tc.Renderer('../output/frames/vaze.png', overwrite=True)
    renderer.initialize(preset='pt', scene=create_scene(), min_path_length=2, max_path_length=3)
    renderer.set_post_processor(tc.post_process.LDRDisplay(adaptive_exposure=False, exposure=1e3, bloom_radius=0.0))
    renderer.render(10000, 20)
