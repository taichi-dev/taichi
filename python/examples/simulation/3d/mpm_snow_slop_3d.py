import math

from taichi.dynamics.mpm import MPM3
from taichi.core import tc_core
from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture
import taichi as tc

gi_render = False
step_number = 1000
# step_number = 1
# total_frames = 1
grid_downsample = 2
output_downsample = 1
render_epoch = 20


def create_snow_block(fn):
    particles = tc_core.RenderParticles()
    assert particles.read(fn)
    downsample = grid_downsample
    tex = Texture.from_render_particles((255 / downsample, 255 / downsample, 255 / downsample), particles) * 5
    # mesh_transform = tc_core.Matrix4(1.0).scale_s(0.5).translate(Vector(0.5, 0.5, 0.5))
    # transform = tc_core.Matrix4(1.0).scale_s(2).scale(Vector(2.0, 0.5, 1.0)).translate(Vector(-2, -0.99, -1))
    mesh_transform = tc_core.Matrix4(1.0).translate(Vector(0, 0.01, 0))
    transform = tc_core.Matrix4(1.0).scale_s(2).translate(Vector(-1, -1, -1))
    vol = VolumeMaterial('sdf_voxel', scattering=5, absorption=0, tex=tex,
                         resolution=(255 / downsample, 255 / downsample, 255 / downsample),
                         transform_ptr=transform.get_ptr_string())
    material = SurfaceMaterial('plain_interface')
    material.set_internal_material(vol)
    return Mesh('cube', material=material, transform=transform * mesh_transform)


def create_sand_scene(frame, d, t):
    downsample = output_downsample
    width, height = 540 / downsample, 540 / downsample
    camera = Camera('thinlens', width=width, height=height, fov=75,
                    origin=(0, 1, 4), look_at=(0.0, -0.9, 0.0), up=(0, 1, 0), aperture=0.01)

    scene = Scene()
    with scene:
        scene.set_camera(camera)
        rep = Texture.create_taichi_wallpaper(10, rotation=0, scale=0.95) * Texture('const', value=(0.7, 0.5, 0.5))
        material = SurfaceMaterial('pbr', diffuse_map=rep)
        scene.add_mesh(Mesh('holder', material=material, translate=(0, -1, -6), scale=2))

        mesh = Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(1.0, 3.0, -1), scale=(0.1, 0.1, 0.1), rotation=(180, 0, 0))
        scene.add_mesh(mesh)

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

    mpm = MPM3(resolution=resolution, gravity=(0, -20, 0), base_delta_t=0.001, num_threads=2)

    levelset = mpm.create_levelset()
    levelset.add_plane(0, 1, 0, -1)
    levelset.add_plane(1, 1, 0, -1.7)
    levelset.global_increase(1)
    tex = Texture('levelset3d', levelset=levelset, bounds=(0, 0.04 / levelset.get_delta_x())) * 6
    tex = Texture('bound', tex=tex, axis=2, bounds=(0.33, 0.66), outside_val=(0, 0, 0))
    tex = Texture('bound', tex=tex, axis=0, bounds=(0.05, 1.0), outside_val=(0, 0, 0))
    mpm.add_particles(density_tex=tex.id, initial_velocity=(0, 0, 0), compression=1.09)
    tex_ball = Texture('sphere', center=(0.12, 0.40, 0.5), radius=0.06) * 10
    mpm.add_particles(density_tex=tex_ball.id, initial_velocity=(0, 0, 0), compression=0.95)

    levelset.set_friction(1)
    mpm.set_levelset(levelset, False)

    t = 0
    for i in range(step_number):
        print 'process(%d/%d)' % (i, step_number)
        mpm.step(0.003)
        tc.core.print_profile_info()
        t += 0.003
        if gi_render:
            d = mpm.get_directory()
            if i % 10 == 0:
                render_sand_frame(i, d, t)
                pass
    mpm.make_video()
