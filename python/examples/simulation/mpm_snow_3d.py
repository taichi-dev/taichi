from taichi.util import *
from taichi.core import tc_core
from taichi.tools.video import VideoManager
from taichi.visual.post_process import LDRDisplay
from taichi.visual.camera import Camera
from taichi.visual.particle_renderer import ParticleRenderer
from taichi.visual.texture import Texture
import math
import random
import time
import cv2

class MPM3:
    def __init__(self, **kwargs):
        self.c = tc_core.create_simulation3d('mpm')
        self.c.initialize(P(**kwargs))
        self.directory = '../output/frames/' + get_uuid() + '/'
        self.video_manager = VideoManager(self.directory, 960, 540)
        try:
            os.mkdir(self.directory)
        except Exception as e:
            print e
        self.particle_renderer = ParticleRenderer('shadow_map',
                                                  shadow_map_resolution=0.5, alpha=0.5, shadowing=0.1, ambient_light=0.2,
                                                  light_direction=(-1, 1, -1))
        self.resolution = kwargs['resolution']
        self.frame = 0

    def step(self, step_t):
        t = self.c.get_current_time()
        print 'Simulation time:', t
        T = time.time()
        self.c.step(step_t)
        print 'Time:', time.time() - T
        image_buffer = tc_core.RGBImageFloat(self.video_manager.width, self.video_manager.height, Vector(0, 0, 0.0))
        particles = self.c.get_render_particles()
        particles.write(self.directory + 'particles%05d.bin' % self.frame)
        res = map(float, self.resolution)
        radius = res[0] * 2.5
        theta = 0
        camera = Camera('pinhole', origin=(0, res[1], res[2] * 5),
                        look_at=(0, 0, 0), up=(0, 1, 0), fov_angle=70,
                        width=10, height=10)
        self.particle_renderer.set_camera(camera)
        self.particle_renderer.render(image_buffer, particles)
        img = image_buffer_to_ndarray(image_buffer)
        img = LDRDisplay(exposure=0.1).process(img)
        cv2.imshow('Vis', img)
        cv2.waitKey(1)
        self.video_manager.write_frame(img)
        self.frame += 1

    def make_video(self):
        self.video_manager.make_video()

if __name__ == '__main__':
    downsample = 1
    resolution = (511 / downsample, 127 / downsample, 255 / downsample)
    tex = Texture('image', filename='D:/assets/taichi_words.png') * 8
    tex = Texture('bounded', tex=tex.id, axis=2, bounds=(0.475, 0.525), outside_val=(0, 0, 0))
    mpm = MPM3(resolution=resolution, gravity=(0, -10, 0), initial_velocity=(0, 0, 0), delta_t=0.001, num_threads=8,
               density_tex=tex.id)
    for i in range(1000):
        mpm.step(0.05)
    mpm.make_video()

