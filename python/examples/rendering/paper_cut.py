from taichi.visual import *
from taichi.util import Vector
from taichi.visual.texture import Texture
from taichi.visual.post_process import *
import math
import random
import colorsys

def create_scene():
    downsample = 1
    width, height = 960 / downsample, 540 / downsample
    camera = Camera('pinhole', width=width, height=height, fov=90,
                    origin=(0, 0, 10), look_at=(0, 0, 0), up=(0, 1, 0))

    scene = Scene()

    with scene:
        scene.set_camera(camera)

        taichi_tex = Texture('taichi', scale=0.96, rotation=math.pi / 2)

        mesh = Mesh('plane', SurfaceMaterial('pbr', diffuse=(.1, .1, .1)),
                    translate=(0, 0, -0.05), scale=10, rotation=(90.3, 0, 0))
        scene.add_mesh(mesh)

        mesh = Mesh('plane', SurfaceMaterial('pbr', diffuse=(0.2, 0.5, 0.2)),
                    translate=(0, 0, 0), scale=(8.3, 1, 4.5), rotation=(90, 0, 0))
        scene.add_mesh(mesh)

        ring_tex = 1 - Texture('ring', inner=0.0, outer=1.0)
        grid_tex = (1 - Texture('rect', bounds=(0.9, 0.9, 1.0))).repeat(6, 6, 1)

        # Taichi circle
        mesh = Mesh('plane', SurfaceMaterial('transparent',
                                             nested=SurfaceMaterial('diffuse', diffuse=(1, 1, 1)),
                                             mask=taichi_tex),
                    translate=(-3.7, 0, 0.05), scale=2, rotation=(90, 0, 0))
        scene.add_mesh(mesh)

        for i in range(1, 5):
            inv_ring_tex = Texture('ring', inner=0.0, outer=0.5 + i * 0.1)
            color = colorsys.hls_to_rgb(i * 0.1, 0.5, 1.0)
            scene.add_mesh(Mesh('plane', SurfaceMaterial('transparent',
                                                     nested=SurfaceMaterial('diffuse', diffuse=color),
                                                     mask=inv_ring_tex),
                            translate=(-3.7, 0, i * 0.03), scale=4, rotation=(90, 0, 0)))

        scene.add_mesh(Mesh('plane', SurfaceMaterial('transparent',
                                                     nested=SurfaceMaterial('diffuse', diffuse=(0, 0.2, 0.5)),
                                                     mask=grid_tex),
                            translate=(4.3, 0, 0.17), scale=1, rotation=(90, 30, 0)))

        scene.add_mesh(Mesh('plane', SurfaceMaterial('transparent',
                                                     nested=SurfaceMaterial('diffuse', diffuse=(1, 1, 0)),
                                                     mask=grid_tex),
                            translate=(4.3, 0, 0.07), scale=2, rotation=(90, 60, 0)))

        scene.add_mesh(Mesh('plane', SurfaceMaterial('transparent',
                                                     nested=SurfaceMaterial('diffuse', diffuse=(0, 1, 1)),
                                                     mask=grid_tex),
                            translate=(4.3, 0, 0.02), scale=3, rotation=(90, 0, 0)))

        mesh = Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(-30, 30, 10), scale=6, rotation=(0, 0, -90))
        scene.add_mesh(mesh)

        mesh = Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(30, 0, 10), scale=2, rotation=(0, 0, 90))
        scene.add_mesh(mesh)

    return scene


if __name__ == '__main__':
    renderer = Renderer('pt', '../output/frames/paper_cut.png', overwrite=True)

    scene = create_scene()
    renderer.set_scene(scene)
    renderer.initialize(min_path_length=1, max_path_length=10,
                        initial_radius=0.5, sampler='sobol', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1,
                        stage_frequency=3, num_threads=8)
    renderer.set_post_processor(LDRDisplay(exposure=1.5, bloom_radius=0.1))
    renderer.render(10000, 20)
