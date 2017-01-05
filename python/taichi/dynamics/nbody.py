import time

import cv2

from taichi.core import tc_core
from taichi.misc.util import *
from taichi.tools.video import VideoManager
from taichi.visual.camera import Camera
from taichi.visual.particle_renderer import ParticleRenderer


class NBody:
    def __init__(self, **kwargs):
        self.c = tc_core.create_simulation3d('nbody')
        self.c.initialize(P(**kwargs))
        self.directory = '../output/frames/' + get_uuid() + '/'
        try:
            os.mkdir(self.directory)
        except Exception as e:
            print e
        self.video_manager = VideoManager(self.directory, 512, 512)
        self.particle_renderer = ParticleRenderer('shadow_map',
                                                  shadow_map_resolution=0.01, alpha=0.12, shadowing=0.1, ambient_light=0.3,
                                                  light_direction=(1, 3, -3))
        self.step_counter = 0

    def step(self, step_t):
        t = self.c.get_current_time()
        print 'Simulation time:', t
        T = time.time()
        self.c.step(step_t)
        print 'Time:', time.time() - T
        image_buffer = tc_core.RGBImageFloat(self.video_manager.width, self.video_manager.height, Vector(0, 0, 0.0))
        particles = self.c.get_render_particles()
        particles.write(self.directory + 'particles%05d.bin' % self.step_counter)
        camera = Camera('pinhole', origin=(0, 0, 50),
                        look_at=(0, 0, 0), up=(0, 1, 0), fov=70,
                        width=self.video_manager.width, height=self.video_manager.height)
        self.particle_renderer.set_camera(camera)
        self.particle_renderer.render(image_buffer, particles)
        img = image_buffer_to_ndarray(image_buffer)
        #img = LDRDisplay(exposure=1, adaptive_exposure=False).process(img)
        cv2.imshow('Vis', img)
        cv2.waitKey(1)
        self.video_manager.write_frame(img)
        self.step_counter += 1

    def make_video(self):
        self.video_manager.make_video()
