import taichi as tc
from taichi.dynamics.mpm import MPM3
from taichi.visual.texture import Texture

if __name__ == '__main__':
    downsample = 3
    resolution = (511 / downsample, 127 / downsample, 255 / downsample)

    mpm = MPM3(resolution=resolution, gravity=(0, -10, 0), base_delta_t=0.001, num_threads=8)

    tex = Texture('image', filename=tc.get_asset_path('textures/taichi_words.png')) * 8
    tex = Texture('bound', tex=tex, axis=2, bounds=(0.475, 0.525), outside_val=(0, 0, 0))
    mpm.add_particles(density_tex=tex.id, initial_velocity=(0, 0, 0))

    for i in range(1000):
        mpm.step(0.05)
    mpm.make_video()

