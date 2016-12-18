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
    camera = Camera('thinlens', width=width, height=height, fov=55,
                    origin=(30, 40, 50), look_at=(0, 0, 0), up=(0, 1, 0), focus=(0, 3, 0), aperture=0.1)

    scene = Scene()

    with scene:
        scene.set_camera(camera)

        # Plane
        scene.add_mesh(Mesh('plane', SurfaceMaterial('pbr', diffuse=(1, 1, 1.0)),
                            translate=(0, 0, 0), scale=100, rotation=(0, 0, 0)))

        for i in range(8):
            menger = 1 - Texture("menger", limit=i)
            scene.add_mesh(Mesh('plane',
                                SurfaceMaterial('transparent', mask=menger,
                                                nested=
                                                SurfaceMaterial('diffuse', diffuse=colorsys.hls_to_rgb(i * 0.1 + 0.3, 0.3, 1.0))),
                                translate=(i * 7 - 28, 3.5, -5), scale=3, rotation=(90, 0, 0)))


        # Lights
        scene.add_mesh(Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(0, 100, -200), scale=5, rotation=(180, 0, 0)))

        scene.add_mesh(Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                            translate=(0, 100, 200), scale=3, rotation=(180, 0, 0)))


    return scene


if __name__ == '__main__':
    renderer = Renderer('pt', '../output/frames/fractals.png', overwrite=True)

    scene = create_scene()
    renderer.set_scene(scene)
    renderer.initialize(min_path_length=1, max_path_length=10,
                        initial_radius=0.5, sampler='sobol', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1,
                        stage_frequency=3, num_threads=8)
    renderer.set_post_processor(LDRDisplay(exposure=1.0, bloom_radius=0.1))
    renderer.render(10000, 20)
