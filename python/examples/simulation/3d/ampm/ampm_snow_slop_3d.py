import math

from taichi.dynamics.mpm import MPM3
from taichi.core import tc_core
from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture
from colorsys import hsv_to_rgb
import taichi as tc

gi_render = False
step_number = 10000
# step_number = 1
# total_frames = 1
grid_downsample = 8
output_downsample = 5
render_epoch = 30


def create_mpm_sand_block(fn):
    particles = tc_core.RenderParticles()
    assert particles.read(fn)
    downsample = grid_downsample
    tex = Texture.from_render_particles((255 / downsample, 255 / downsample, 255 / downsample), particles) * 5
    # tex = Texture('sphere', center=(0.5, 0.5, 0.5), radius=0.5)
    with tc.transform_scope(scale=2):
        return tc.create_volumetric_block(tex, res=(128, 128, 128))


def create_scene(frame, d, t):
    downsample = output_downsample
    width, height = 1280 / downsample, 720 / downsample

    camera = Camera('pinhole', width=width, height=height, fov=30,
                    origin=(0, 0, 8), look_at=(0, 0, 0), up=(0, 1, 0))

    scene = Scene()
    with scene:
        scene.set_camera(camera)
        mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(30000, 40000, 60000)),
                       translate=(-20, 20, -15), scale=3, rotation=(0, 0, 180))
        scene.add_mesh(mesh)

        with tc.transform_scope(rotation=(10, 0, 0), translate=(0, 0, 0), scale=1):
            with tc.transform_scope(rotation=(0, -40, 0), translate=(0, 0.5, 0), scale=1):
                fn = d + r'/particles%05d.bin' % frame
                mesh = create_mpm_sand_block(fn)
                scene.add_mesh(mesh)

        envmap_texture = Texture('spherical_gradient', inside_val=(10, 10, 10, 10), outside_val=(1, 1, 1, 0),
                                 angle=10, sharpness=20)
        envmap = EnvironmentMap('base', texture=envmap_texture.id, res=(1024, 1024))
        scene.set_environment_map(envmap)

    return scene


def render_frame(frame, d, t):
    renderer = Renderer(output_dir='volumetric', overwrite=True, frame=frame)
    renderer.initialize(preset='pt', scene=create_scene(frame, d, t), sampler='prand', max_path_length=4)
    renderer.set_post_processor(LDRDisplay(exposure=1, bloom_radius=0.00, bloom_threshold=1.0))
    renderer.render(render_epoch)


if __name__ == '__main__':
    downsample = grid_downsample
    resolution = (255 / downsample, 255 / downsample, 255 / downsample)
    print resolution
    frame_dt = 0.1

    # mpm = MPM3(resolution=resolution, gravity=(0, -40, 0), async=True, num_threads=4, strength_dt_mul=2, base_delta_t=1e-7)
    mpm = MPM3(resolution=resolution, gravity=(0, -40, 0), async=False, num_threads=4, base_delta_t=0.0004)

    levelset = mpm.create_levelset()
    height_ = 0.0
    ground_ = 30.0
    half_ = (180.0 - ground_) / 2
    norm_ = 90.0 - ground_
    cross_x = 0.75 + height_ / math.tan(half_ / 180 * math.pi)
    cross_y = 0 + height_
    cos_ = math.cos(norm_ / 180 * math.pi)
    sin_ = math.sin(norm_ / 180 * math.pi)
    levelset.add_plane(0, 1, 0, -cross_y)
    levelset.add_plane(cos_, sin_, 0, -cross_x * cos_ - cross_y * sin_)
    levelset.global_increase(height_)

    tex = Texture('levelset3d', levelset=levelset, bounds=(0, 0.02 / levelset.get_delta_x())) * 2
    tex = Texture('bound', tex=tex, axis=2, bounds=(0.22, 0.78), outside_val=(0, 0, 0))
    tex = Texture('bound', tex=tex, axis=0, bounds=(0.05, 1.0), outside_val=(0, 0, 0))
    mpm.add_particles(density_tex=tex.id, initial_velocity=(0, 0, 0), compression=1.15, lambda_0=3000, mu_0=3000)
    tex_ball = Texture('sphere', center=(0.11, 0.51, 0.5), radius=0.08) * 3
    mpm.add_particles(density_tex=tex_ball.id, initial_velocity=(0, 0, 0), compression=0.95)

    levelset.set_friction(1)
    mpm.set_levelset(levelset, False)

    t = 0
    for i in range(step_number):
        print 'process(%d/%d)' % (i, step_number)
        camera = Camera('pinhole', origin=(resolution[0] * 1.08, resolution[1] * -0.1, resolution[2] * 1.12),
                        look_at=(0, -resolution[1] * 0.5, 0), up=(0, 1, 0), fov=90,
                        width=10, height=10)
        mpm.step(frame_dt, camera=camera)
        t += 0.01
        if gi_render:
            d = mpm.get_directory()
            if i % 8 == 0:
                render_frame(i, d, t)
                pass
    mpm.make_video()
