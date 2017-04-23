import math

from taichi.dynamics.mpm import MPM3
from taichi.core import tc_core
from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture
from colorsys import hsv_to_rgb
import taichi as tc

gi_render = True
step_number = 400
# step_number = 1
# total_frames = 1
grid_downsample = 2
output_downsample = 2
render_epoch = 20


def create_mpm_sand_block(fn):
    particles = tc_core.RenderParticles()
    assert particles.read(fn)
    downsample = grid_downsample
    tex = Texture.from_render_particles((255 / downsample, 255 / downsample, 255 / downsample), particles) * 5
    # mesh_transform = tc_core.Matrix4(1.0).scale_s(0.5).translate(Vector(0.5, 0.5, 0.5))
    # transform = tc_core.Matrix4(1.0).scale_s(2).scale(Vector(2.0, 0.5, 1.0)).translate(Vector(-2, -0.99, -1))
    mesh_transform = tc_core.Matrix4(1.0).translate(Vector(0, 0.01, 0))
    transform = tc_core.Matrix4(1.0).scale_s(1).translate(Vector(-1, -1, -1))
    vol = VolumeMaterial('sdf_voxel', scattering=5, absorption=0, tex=tex,
                         resolution=(255 / downsample, 255 / downsample, 255 / downsample),
                         transform_ptr=transform.get_ptr_string())
    material = SurfaceMaterial('plain_interface')
    material.set_internal_material(vol)
    return Mesh('cube', material=material, transform=transform * mesh_transform)


def create_sand_scene(frame, d, t):
    downsample = output_downsample
    width, height = 1280 / downsample, 720 / downsample
    camera = Camera('pinhole', width=width, height=height, fov=30,
                    origin=(0, 0, 10), look_at=(0, 0, 0), up=(0, 1, 0))

    scene = Scene()
    with scene:
        scene.set_camera(camera)
        tex = Texture.create_taichi_wallpaper(20, rotation=0, scale=0.95) * 0.9

        emission = 100000
        mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(emission, emission, emission)),
                       translate=(300, 200, 300), scale=30, rotation=(0, 0, 180))
        scene.add_mesh(mesh)

        with tc.transform_scope(rotation=(0, 0, 0), scale=0.8):
            material = SurfaceMaterial('diffuse', color=(1, 0.7, 1), roughness_map=tex.id, f0=1)
            #scene.add_mesh(Mesh('cube', material=material, translate=(0, -1, 0), scale=(2, 0.02, 1)))
            for i in range(0):
                material = SurfaceMaterial('diffuse', color=hsv_to_rgb(i * 0.2, 0.5, 1.0), roughness_map=tex.id, f0=1)
                scene.add_mesh(
                    Mesh('cube', material=material, translate=(2, 0.3 * (i - 3), 0.2), scale=(0.01, 0.10, 0.5)))
            material = SurfaceMaterial('diffuse', color=(1, 1, 1), roughness_map=tex.id, f0=1)
            scene.add_mesh(Mesh('cube', material=material, translate=(0, 0, -1.1), scale=(1.9, 0.9, 0.03)))

        envmap_texture = Texture('spherical_gradient', inside_val=(10, 10, 10, 10), outside_val=(1, 1, 1, 0),
                                 angle=10, sharpness=20)
        envmap = EnvironmentMap('base', texture=envmap_texture.id, res=(1024, 1024))
        # scene.set_environment_map(envmap)

        # Change this line to your particle output path pls.
        # fn = r'../sand-sim/particles%05d.bin' % frame
        fn = d + r'/particles%05d.bin' % frame
        mesh = create_mpm_sand_block(fn)
        scene.add_mesh(mesh)

    return scene


def render_sand_frame(frame, d, t):
    renderer = Renderer(output_dir='volumetric', overwrite=True, frame=frame)
    renderer.initialize(preset='pt', scene=create_sand_scene(frame, d, t), sampler='prand')
    renderer.set_post_processor(LDRDisplay(exposure=0.6, bloom_radius=0.0, bloom_threshold=1.0))
    renderer.render(render_epoch)


if __name__ == '__main__':
    downsample = grid_downsample
    resolution = (255 / downsample, 255 / downsample, 255 / downsample)

    mpm = MPM3(resolution=resolution, gravity=(0, -20, 0), async=True, num_threads=8, strength_dt_mul=4)

    tex = Texture('mesh', resolution=resolution, filename=tc.get_asset_path('meshes/bunny.obj')) * 8

    # bug1
    tex = tex.zoom((0.3, 0.3, 0.3), (0.5, 0.6, 0.5), False)
    # bug2
    # tex = tex.zoom((0.3, 0.3, 0.3), (0.5, 0.0, 0.5), False)

    mpm.add_particles(density_tex=tex.id, initial_velocity=(0, 0, 0))

    t = 0
    for i in range(step_number):
        print 'process(%d/%d)' % (i, step_number)
        mpm.step(0.00)
        t += 0.00
        if gi_render:
            d = mpm.get_directory()
            if i % 10 == 0:
                render_sand_frame(i, d, t)
                pass
    mpm.make_video()
