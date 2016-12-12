from taichi.visual import *
from taichi.util import Vector
from taichi.visual.texture import Texture
from taichi.visual.post_process import *

def create_scene():
    downsample = 1
    width, height = 960 / downsample, 540 / downsample
    camera = Camera('perspective', width=width, height=height, fov_angle=50,
                    origin=(1, 1, 3), look_at=(0, -0.6, 0), up=(0, 1, 0))

    scene = Scene()

    dist = 100

    with scene:
        scene.set_camera(camera)
        rep = Texture.create_taichi_wallpaper(10, rotation=0, scale=0.95) * Texture('const', value=(0.5, 0.5, 1.0))
        material = SurfaceMaterial('pbr', diffuse_map=rep.id)
        scene.add_mesh(Mesh('holder', material=material, translate=(0, -1.2, -5), scale=2))

        mesh = Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(-0.5 * dist, 0.7 * dist, 0.4 * dist), scale=0.01, rotation=(180, 0, 0))
        scene.add_mesh(mesh)

        mesh = Mesh('cube', SurfaceMaterial(
            'pbr', color=(1, 1, 1), specular=(1, 1, 1), transparent=True, ior=2.5, glossiness=-1),
                    translate=(0, -0.5, 0), scale=0.4)
        scene.add_mesh(mesh)
        mesh = Mesh('torus', SurfaceMaterial(
            'pbr', diffuse=(1, 0.5, 0)),
                    translate=(0, -0.5, 0), scale=0.25, rotation=(90, 0, 0))
        scene.add_mesh(mesh)


    return scene

if __name__ == '__main__':
    renderer = Renderer('sppm', '../output/frames/amcmcups.png', overwrite=True)

    scene = create_scene()
    renderer.set_scene(scene)
    renderer.initialize(min_path_length=1, max_path_length=10,
                        initial_radius=0.005, sampler='prand', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1, stage_frequency=3,
                        num_threads=8)
    renderer.set_post_processor(LDRDisplay(exposure=0.3, bloom_radius=0.00))
    renderer.render(10000, 20)
