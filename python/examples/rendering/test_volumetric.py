from taichi.visual import *
from taichi.util import Vector
from taichi.visual.texture import Texture
from taichi.visual.post_process import *
from taichi.core import tc_core
import math

def create_taichi_scene(eye_position):
    downsample = 2
    width, height = 960 / downsample, 540 / downsample
    camera = Camera('pinhole', width=width, height=height, fov=70,
                    origin=(0.5, 0.5, 3), look_at=(0.5, 0.5, 0.5), up=(0, 1, 0))

    scene = Scene()
    with scene:
        scene.set_camera(camera)
        rep = Texture.create_taichi_wallpaper(10, rotation=0, scale=0.95) * Texture('const', value=(0.5, 0.5, 1.0))
        material = SurfaceMaterial('pbr', diffuse_map=rep)
        scene.add_mesh(Mesh('holder', material=material, translate=(0, -1, -7), scale=2))

        mesh = Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(0.5, 1.3, 0), scale=(0.1, 1.0, 0.1), rotation=(180, 0, 0))
        scene.add_mesh(mesh)

        material = SurfaceMaterial('plain_interface')
        #vol = VolumeMaterial("homogeneous", scattering=10, absorption=0)
        tex = 1 - Texture('taichi', scale=0.95)
        vol = VolumeMaterial('voxel', scattering=100, absorption=0, resolution=256, tex=tex)
        material.set_internal_material(vol)
        mesh = Mesh('cube', material=material,
                    translate=(0.5, 0.5, 0.5), scale=(0.5, 0.5, 0.2), rotation=(0, 0, 0))
        scene.add_mesh(mesh)

        #envmap = EnvironmentMap('base', filepath='d:/assets/schoenbrunn-front_hd.hdr')
        #scene.set_environment_map(envmap)

    return scene

def create_snow_scene(eye_position):
    downsample = 2
    width, height = 960 / downsample, 540 / downsample
    camera = Camera('pinhole', width=width, height=height, fov=70,
                    origin=(0.5, 0.5, 3), look_at=(0.5, 0.5, 0.5), up=(0, 1, 0))

    scene = Scene()
    with scene:
        scene.set_camera(camera)
        rep = Texture.create_taichi_wallpaper(10, rotation=0, scale=0.95) * Texture('const', value=(0.5, 0.5, 1.0))
        material = SurfaceMaterial('pbr', diffuse_map=rep)
        scene.add_mesh(Mesh('holder', material=material, translate=(0, -1, -7), scale=2))

        mesh = Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(0.5, 1.3, 0), scale=(0.1, 1.0, 0.1), rotation=(180, 0, 0))
        scene.add_mesh(mesh)

        material = SurfaceMaterial('plain_interface')
        #vol = VolumeMaterial("homogeneous", scattering=10, absorption=0)
        particles = tc_core.RenderParticles()
        particles.read('../output/frames/snow-taichi-g10/particles%05d.bin' % 0)
        tex = Texture.from_render_particles((511, 127, 255), particles)
        vol = VolumeMaterial('voxel', scattering=100, absorption=0, resolution=256, tex=tex)
        material.set_internal_material(vol)
        mesh = Mesh('cube', material=material,
                    translate=(0.5, 0.5, 0.5), scale=(0.5, 0.5, 0.5), rotation=(0, 0, 0))
        scene.add_mesh(mesh)

        #envmap = EnvironmentMap('base', filepath='d:/assets/schoenbrunn-front_hd.hdr')
        #scene.set_environment_map(envmap)

    return scene

if __name__ == '__main__':
    renderer = Renderer('pt', '../output/frames/volumetric.png', overwrite=True)

    eye_position = Vector(0.9, -0.3)
    scene = create_snow_scene(eye_position)
    renderer.set_scene(scene)
    renderer.initialize(min_path_length=1, max_path_length=10,
                        initial_radius=0.005, sampler='prand', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1, stage_frequency=3,
                        num_threads=8)
    renderer.set_post_processor(LDRDisplay(exposure=3, bloom_radius=0.00))
    renderer.render(1000000, 1000)
