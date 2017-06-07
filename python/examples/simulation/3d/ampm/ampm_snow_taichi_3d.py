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
step_number = 100000
# step_number = 1
# total_frames = 1
grid_downsample = 4
output_downsample = 1
render_epoch = 33


def create_mpm_snow_block(fn):
    particles = tc_core.RenderParticles()
    assert particles.read(fn)
    downsample = grid_downsample
    tex = Texture.from_render_particles((511 / downsample, 127 / downsample, 255 / downsample), particles) * 5
    # tex = Texture('sphere', center=(0.5, 0.5, 0.5), radius=0.5)
    with tc.transform_scope(translate=(0, -0.75, 0), scale=(2, 0.5, 1)):
        return tc.create_volumetric_block(tex, res=(256, 256, 256))


def create_scene(frame, d, t):
    downsample = output_downsample
    width, height = 1280 / downsample, 720 / downsample

    camera = Camera('pinhole', width=width, height=height, fov=25,
                    origin=(0, 0, 6), look_at=(0, 0, 0), up=(0, 1, 0))
    # camera = Camera('pinhole', width=width, height=height, fov=30,
    #                 origin=(2, 4, 4), look_at=(0, 0, 0), up=(0, 1, 0))

    scene = Scene()
    with scene:
        scene.set_camera(camera)

        with tc.transform_scope(rotation=(20, 0, 0), translate=(0, 0.75, 0), scale=1):
            mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(30000, 40000, 60000)),
                           translate=(-20, 30, 0), scale=3, rotation=(0, 0, 180))
            scene.add_mesh(mesh)
            mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(6000, 8000, 12000)),
                           translate=(20, 30, 40), scale=3, rotation=(0, 0, -180))
            scene.add_mesh(mesh)

            material = SurfaceMaterial('diffuse', color=(0.24, 0.18, 0.12), f0=1)
            scene.add_mesh(Mesh('cube', material=material, translate=(0, -1.01, 0), scale=(1, 0.02, 0.6)))

            fn = d + r'/particles%05d.bin' % frame
            mesh = create_mpm_snow_block(fn)
            scene.add_mesh(mesh)

        envmap_texture = Texture('spherical_gradient', inside_val=(10, 10, 10, 10), outside_val=(1, 1, 1, 0),
                                 angle=10, sharpness=20)
        envmap = EnvironmentMap('base', texture=envmap_texture.id, res=(1024, 1024))
        scene.set_environment_map(envmap)

    return scene


def render_frame(frame, d, t):
    renderer = Renderer(output_dir='volumetric', overwrite=True, frame=frame)
    renderer.initialize(preset='pt', scene=create_scene(frame, d, t), sampler='prand', max_path_length=3)
    renderer.set_post_processor(LDRDisplay(exposure=1, bloom_radius=0.00, bloom_threshold=1.0))
    renderer.render(render_epoch)


if __name__ == '__main__':
    downsample = grid_downsample
    resolution = (511 / downsample, 127 / downsample, 255 / downsample)

    mpm = MPM3(resolution=resolution, gravity=(0, -20, 0), async=True, num_threads=8, strength_dt_mul=4)

    tex = Texture('image', filename=tc.get_asset_path('textures/taichi_words.png')) * 8
    tex = Texture('bound', tex=tex, axis=2, bounds=(0.475, 0.525), outside_val=(0, 0, 0))
    # tex = tex * (Texture('perlin').zoom((16, 16, 16), (0.1, 0.2, 0.3)) * 5 + 3)
    mpm.add_particles(density_tex=tex.id, initial_velocity=(0, 0, 0))

    # levelset = mpm.create_levelset()
    # levelset.add_cuboid((0.01, 0.01, 0.01), (0.99, 0.99, 0.99), True)
    # mpm.set_levelset(levelset)

    t = 0
    for i in range(step_number):
        print 'process(%d/%d)' % (i, step_number)
        camera = Camera('pinhole', origin=(0, resolution[1] * 0.4, resolution[2] * 2.4),
                        look_at=(0, -resolution[1] * 0.5, 0), up=(0, 1, 0), fov=90,
                        width=10, height=10)
        mpm.step(0.01, camera=camera)
        t += 0.01
        if gi_render:
            d = mpm.get_directory()
            if i % 40 == 0:
                render_frame(i, d, t)
                pass
    mpm.make_video()
