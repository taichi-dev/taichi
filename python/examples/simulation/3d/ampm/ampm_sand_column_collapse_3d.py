from taichi.dynamics.mpm import MPM3
from taichi.core import tc_core
from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture

gi_render = False
step_number = 200
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
    mesh_transform = tc_core.Matrix4(1.0).scale_s(0.5).translate(Vector(0.5, 0.5, 0.5))
    transform = tc_core.Matrix4(1.0).scale_s(2).scale(Vector(2.0, 0.5, 1.0)).translate(Vector(-2, -0.99, -1))
    vol = VolumeMaterial('sdf_voxel', scattering=5, absorption=0, tex=tex,
                         resolution=(255 / downsample, 255 / downsample, 255 / downsample),
                         transform_ptr=transform.get_ptr_string())
    material = SurfaceMaterial('plain_interface')
    material.set_internal_material(vol)
    return Mesh('cube', material=material, transform=transform * mesh_transform)


def create_sand_scene(frame, d):
    downsample = output_downsample
    width, height = 540 / downsample, 540 / downsample
    camera = Camera('thinlens', width=width, height=height, fov=35,
                    origin=(0, 1, 4), look_at=(0.0, -0.9, -0.0), up=(0, 1, 0), aperture=0.08)

    scene = Scene()
    with scene:
        scene.set_camera(camera)
        rep = Texture.create_taichi_wallpaper(10, rotation=0, scale=0.95) * Texture('const', value=(0.7, 0.5, 0.5))
        material = SurfaceMaterial('pbr', diffuse_map=rep)
        scene.add_mesh(Mesh('holder', material=material, translate=(0, -1, -6), scale=2))

        mesh = Mesh('plane', SurfaceMaterial('emissive', color=(1, 1, 1)),
                    translate=(1.0, 1.0, -1), scale=(0.1, 0.1, 0.1), rotation=(180, 0, 0))
        scene.add_mesh(mesh)

        # Change this line to your particle output path pls.
        # fn = r'../sand-sim/particles%05d.bin' % frame
        fn = d + r'/particles%05d.bin' % frame
        mesh = create_mpm_sand_block(fn)
        scene.add_mesh(mesh)

    return scene


def render_sand_frame(frame, d):
    renderer = Renderer(output_dir='volumetric', overwrite=True, frame=frame)
    renderer.initialize(preset='pt', scene=create_sand_scene(frame, d), sampler='prand')
    renderer.set_post_processor(LDRDisplay(exposure=0.6, bloom_radius=0.0, bloom_threshold=1.0))
    renderer.render(render_epoch)


if __name__ == '__main__':
    downsample = grid_downsample
    resolution = (432 / downsample, 144 / downsample, 432 / downsample)
    tex = Texture('ring', outer=0.15) * 2
    tex = Texture('bound', tex=tex, axis=2, bounds=(0.0, 0.8), outside_val=(0, 0, 0))
    tex = Texture('rotate', tex=tex, rotate_axis=0, rotate_times=1)
    mpm = MPM3(resolution=resolution, gravity=(0, -100, 0), base_delta_t=0.000001, num_threads=8, ampm=True)
    mpm.add_particles(type="dp", density_tex=tex.id, initial_velocity=(0, 0, 0))
    for i in range(step_number):
        print 'process(%d/%d)' % (i, step_number)
        mpm.step(0.01)
        if gi_render:
            d = mpm.get_directory()
            if i % 20 == 0:
                render_sand_frame(i, d)
    mpm.make_video()
