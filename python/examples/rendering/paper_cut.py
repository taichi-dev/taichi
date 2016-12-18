from taichi.visual import *
from taichi.util import Vector
from taichi.visual.texture import Texture
from taichi.visual.post_process import *
import random
import colorsys


def create_scene():
    downsample = 1
    width, height = 960 / downsample, 540 / downsample
    camera = Camera('pinhole', width=width, height=height, fov_angle=90,
                    origin=(0, 0, 10), look_at=(0, 0, 0), up=(0, 1, 0))

    scene = Scene()

    with scene:
        scene.set_camera(camera)

        texture = (Texture('perlin') + 1) * 1 + 0.01
        texture = Texture('fract', tex=texture)

        mesh = Mesh('plane', SurfaceMaterial('pbr', diffuse_map=texture.id),
                    translate=(0, 0, -0.05), scale=10, rotation=(90, 0, 0))
        scene.add_mesh(mesh)

        mesh = Mesh('plane', SurfaceMaterial('difftrans', diffuse=(1, 1, 1)),
                    translate=(-4, 0, 0), scale=2, rotation=(0, 0, 90))
        scene.add_mesh(mesh)

        mesh = Mesh('plane', SurfaceMaterial('diffuse', diffuse=(0.1, 0.08, 0.08)),
                    translate=(-10, 0, 0), scale=10, rotation=(0, 0, 90))
        scene.add_mesh(mesh)

        taichi_tex = Texture('taichi', scale=0.92)

        mesh = Mesh('plane', SurfaceMaterial('transparent',
                                             nested=SurfaceMaterial('diffuse', diffuse=(0.1, 0.08, 0.08)),
                                             mask=taichi_tex),
                    translate=(3, 0, 1), scale=1, rotation=(0, 0, 90))
        scene.add_mesh(mesh)

        mesh = Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(10, 0, 4), scale=0.1, rotation=(0, 0, 90))
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
    renderer.set_post_processor(LDRDisplay(exposure=0.6, bloom_radius=0.1))
    renderer.render(10000, 20)
