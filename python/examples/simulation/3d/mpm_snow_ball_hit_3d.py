import math

from taichi.dynamics.mpm import MPM3
from taichi.core import tc_core
from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture
import taichi as tc

gi_render = True
step_number = 400
# step_number = 1
# total_frames = 1
grid_downsample = 2
output_downsample = 1
render_epoch = 20


def create_mpm_sand_block(fn):
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
                    translate=(1.0, 1.0, -1), scale=(0.1, 0.1, 0.1), rotation=(180, 0, 0))
        scene.add_mesh(mesh)

        material = tc.SurfaceMaterial('microfacet', color=(1, 1, 0.5), roughness=(0.1, 0, 0, 0), f0=1)
        sphere = tc.Mesh('sphere', material,
                         translate=((t+0.05) * 0.5 - 0.35, -0.61, 0), scale=0.1, rotation=(0, 0, 0))
        scene.add_mesh(sphere)

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

    mpm = MPM3(resolution=resolution, gravity=(0, -10, 0), delta_t=0.0005, num_threads=8)

    # tex = Texture('ring', outer=0.15) * 4
    # tex = Texture('bound', tex=tex, axis=2, bounds=(0.0, 0.4), outside_val=(0, 0, 0))
    # tex = Texture('rotate', tex=tex, rotate_axis=0, rotate_times=1)
    tex = Texture('mesh', resolution=resolution, filename=tc.get_asset_path('meshes/suzanne.obj')) * 8
    tex = tex.zoom((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
    # tex = Texture('rotate', tex=tex, rotate_axis=1, rotate_times=1)
    mpm.add_particles(density_tex=tex.id, initial_velocity=(0, 0, 0))

    # Dynamic Levelset
    def levelset_generator(t):
        levelset = mpm.create_levelset()
        levelset.add_sphere(Vector(0.325 + 0.25 * (t+0.05), 0.2, 0.5), 0, False)
        # levelset.add_sphere(Vector(0.5, 0.2, 0.5), t, False)
        return levelset
    mpm.set_levelset(levelset_generator, True)

    t = 0
    for i in range(step_number):
        print 'process(%d/%d)' % (i, step_number)
        mpm.step(0.01)
        t += 0.01
        if gi_render:
            d = mpm.get_directory()
            if i % 10 == 0:
                render_sand_frame(i, d, t)
                pass
    mpm.make_video()
