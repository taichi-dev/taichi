from taichi.util import *
from taichi.core import tc_core
from taichi.tools.video import VideoManager
from taichi.visual.post_process import LDRDisplay
from taichi.visual.camera import Camera
from taichi.visual.particle_renderer import ParticleRenderer
import math
import random
import time
import cv2

class MPM3:
    def __init__(self, **kwargs):
        self.c = tc_core.create_simulation3d('mpm')
        self.c.initialize(P(**kwargs))
        self.video_manager = VideoManager(get_uuid(), 480, 270)
        self.particle_renderer = ParticleRenderer('shadow_map',
                                                  shadow_map_resolution=0.5, alpha=0.5, shadowing=0.1, ambient_light=0.2,
                                                  light_direction=(-1, 1, -1))
        self.resolution = kwargs['resolution']

    def step(self, step_t):
        t = self.c.get_current_time()
        print 'Simulation time:', t
        T = time.time()
        self.c.step(step_t)
        print 'Time:', time.time() - T
        image_buffer = tc_core.RGBImageFloat(self.video_manager.width, self.video_manager.height, Vector(0, 0, 0.0))
        particles = self.c.get_render_particles()
        res = map(float, self.resolution)
        radius = res[0] * 2.5
        theta = t * 0.6 + 1.7
        camera = Camera('perspective', origin=(radius * math.cos(theta), radius * 0.3, radius * math.sin(theta)),
                        look_at=(0, 0, 0), up=(0, 1, 0), fov_angle=50,
                        width=10, height=10)
        self.particle_renderer.set_camera(camera)
        self.particle_renderer.render(image_buffer, particles)
        img = image_buffer_to_ndarray(image_buffer)
        img = LDRDisplay(exposure=0.1).process(img)
        cv2.imshow('Vis', img)
        cv2.waitKey(1)
        self.video_manager.write_frame(img)

    def make_video(self):
        self.video_manager.make_video()

if __name__ == '__main__':
    resolution = (64, 64, 32)
    mpm = MPM3(resolution=resolution,
               gravity=(0, -10, 0), initial_velocity=(0, -30, 0), delta_t=0.002, num_threads=8)
    for i in range(500):
        mpm.step(0.05)

    mpm.make_video()

