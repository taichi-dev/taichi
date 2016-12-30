from math import *

import taichi as tc
from taichi.mics.util import *


def create_scene():
    downsample = 1
    width, height = 960 / downsample, 540 / downsample
    camera = tc.Camera('thinlens', width=width, height=height, fov=40,
                       origin=(0, 20, 40), look_at=(0, 1, 0), up=(0, 1, 0), aperture=0.3)

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
            scene.add_mesh(tc.Mesh(base_mesh, mat, translate=((i - (len(materials) - 1) / 2) * 3, 1.3, 0), scale=1))

        # Ground
        tex = (((tc.Texture('perlin') + 1) * 6).zoom((0.6, 0.6, 0.6))).fract() * (1.0, 0.7, 0.4)
        scene.add_mesh(tc.Mesh('plane', tc.SurfaceMaterial('pbr', diffuse_map=tex), scale=200,
                               translate=(0, 0, 0), rotation=(0, 0, 0)))

        # Board
        gradient = tc.Texture('uv', coeff_u=1, coeff_v=0)

        scene.add_mesh(tc.Mesh('plane', tc.SurfaceMaterial('pbr', diffuse_map=gradient, specular_map=1-gradient), scale=(10, 1, 1),
                               translate=(0, 0.3, 2), rotation=(0, 0, 0)))

        scene.add_mesh(tc.Mesh('plane', tc.SurfaceMaterial('glossy', color=(1, 1, 1), glossiness_map=gradient * 100), scale=(10, 1, 1),
                               translate=(0, 0.3, 4.5), rotation=(0, 0, 0)))

        scene.add_mesh(tc.Mesh('plane', tc.SurfaceMaterial('pbr', diffuse_map=gradient * (1, 0, 0)+(1 - gradient) * (0, 1, 1)), scale=(10, 1, 1),
                               translate=(0, 0.3, 7), rotation=(0, 0, 0)))

        scene.add_mesh(tc.Mesh('plane', tc.SurfaceMaterial('transparent', mask=gradient,
                                                           nested=tc.SurfaceMaterial('diffuse', color=(1, 1, 1))), scale=(10, 1, 1),
                               translate=(0, 0.3, 9.5), rotation=(0, 0, 0)))

        for i in range(10):
            scene.add_mesh(tc.Mesh(tc.geometry.create_mesh_from_functions((50, 50),
                                                                          lambda p: Vector(p.x * 2 - 1,
                                                                                           sin(p.x * 10 * pi) + cos(p.y * 5 * pi), p.y * 2 - 1)),
                                   material=tc.SurfaceMaterial('reflective', color=(1, 1, 1)), translate=(0, 1, -6), scale=(8, 0.2, 2)))

        envmap = tc.EnvironmentMap('base', filepath='../../taichi_assets/envmaps/schoenbrunn-front_hd.hdr')
        envmap.set_transform(tc.core.Matrix4(1.0).rotate_euler(Vector(0, 30, 0)))
        scene.set_environment_map(envmap)

    return scene


if __name__ == '__main__':
    renderer = tc.Renderer('../output/frames/geometry.png', overwrite=True)
    renderer.initialize(preset='pt', scene=create_scene(), luminance_clamping=1000)
    renderer.set_post_processor(tc.post_process.LDRDisplay(exposure=1, bloom_radius=0.0))
    renderer.render(10000, 20)
