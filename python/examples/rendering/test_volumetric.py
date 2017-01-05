from taichi.core import tc_core
from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture


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

def create_mpm_snow_block(fn):
    particles = tc_core.RenderParticles()
    particles.read(fn)
    tex = Texture.from_render_particles((511, 127, 255), particles) * 5
    mesh_transform = tc_core.Matrix4(1.0).scale(Vector(0.5, 0.5, 0.5)).translate(Vector(0.5, 0.5, 0.5))
    transform = tc_core.Matrix4(1.0).scale_s(2).scale(Vector(2.0, 0.5, 1.0)).translate(Vector(-2, -0.99, -1))
    vol = VolumeMaterial('sdf_voxel', scattering=5, absorption=0, tex=tex, resolution=(511, 127, 255),
                         transform_ptr=transform.get_ptr_string())
    material = SurfaceMaterial('plain_interface')
    material.set_internal_material(vol)
    return Mesh('cube', material=material, transform=transform * mesh_transform)

def create_snow_scene(frame):
    downsample = 1
    width, height = 960 / downsample, 540 / downsample
    camera = Camera('thinlens', width=width, height=height, fov=60,
                    origin=(0, 1, 4), look_at=(0.0, -0.7, -0.0), up=(0, 1, 0), aperture=0.05)

    scene = Scene()
    with scene:
        scene.set_camera(camera)
        rep = Texture.create_taichi_wallpaper(10, rotation=0, scale=0.95) * Texture('const', value=(0.7, 0.5, 0.5))
        material = SurfaceMaterial('pbr', diffuse_map=rep)
        scene.add_mesh(Mesh('holder', material=material, translate=(0, -1, -6), scale=2))

        mesh = Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(1.0, 1.0, -1), scale=(0.1, 0.1, 0.1), rotation=(180, 0, 0))
        scene.add_mesh(mesh)

        fn = '../output/frames/snow-taichi-g10/particles%05d.bin' % frame
        mesh = create_mpm_snow_block(fn)
        scene.add_mesh(mesh)

        #envmap = EnvironmentMap('base', filepath='d:/assets/schoenbrunn-front_hd.hdr')
        #scene.set_environment_map(envmap)

    return scene

def render_snow_frame(frame):
    renderer = Renderer('pt', '../output/frames/volumetric', overwrite=True, frame=frame)
    scene = create_snow_scene(frame)
    renderer.set_scene(scene)
    renderer.initialize(min_path_length=1, max_path_length=20,
                        initial_radius=0.005, sampler='prand', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1, stage_frequency=3,
                        num_threads=8, luminance_clamping=0)
    renderer.set_post_processor(LDRDisplay(exposure=0.6, bloom_radius=0.0, bloom_threshold=1.0))
    renderer.render(20)

if __name__ == '__main__':
    total_frames = 550
    for i in range(0, total_frames, 20):
        render_snow_frame(i)
