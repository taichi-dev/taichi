import taichi as tc
import math
import random
import colorsys


def create_scene():
    downsample = 1
    width, height = 960 / downsample, 540 / downsample
    camera = tc.Camera('pinhole', width=width, height=height, fov=90,
                    origin=(0, 0, 10), look_at=(0, 0, 0), up=(0, 1, 0))

    scene = tc.Scene()

    with scene:
        scene.set_camera(camera)

        taichi_tex = tc.Texture('taichi', scale=0.96, rotation=math.pi / 2)

        for i in range(3):
            with tc.TransformScope(translate=(i, 0, 0)):
                for j in range(3):
                    with tc.TransformScope(translate=(0, j, 0)):
                        mesh = tc.Mesh('plane', tc.SurfaceMaterial('pbr', diffuse=(.1, .1, .1)),
                                    translate=(0, 0, -0.05), scale=0.4, rotation=(90.3, 0, 0))
                        scene.add_mesh(mesh)

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(-30, 30, 10), scale=2, rotation=(0, 0, -90))

        scene.add_mesh(mesh)

    return scene


if __name__ == '__main__':
    renderer = tc.Renderer('pt', '../output/frames/paper_cut.png', overwrite=True)

    scene = create_scene()
    renderer.set_scene(scene)
    renderer.initialize(min_path_length=1, max_path_length=5,
                        initial_radius=0.5, sampler='sobol', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1,
                        stage_frequency=3, num_threads=8, shrinking_radius=True)
    renderer.set_post_processor(tc.post_process.LDRDisplay(exposure=0.5, bloom_radius=0.0))
    renderer.render(10000, 20)
