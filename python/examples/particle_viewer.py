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

class ParticleViewer:
    def __init__(self, directory, width, height):
        self.directory = '../output/frames/' + directory + '/'
        self.video_manager = VideoManager(self.directory, width, height)
        self.particle_renderer = ParticleRenderer('shadow_map',
                                                  shadow_map_resolution=0.01, alpha=0.12, shadowing=0.1, ambient_light=0.3,
                                                  light_direction=(1, 3, -3))
        self.step_counter = 0

    def view(self, frame):
        particles = tc_core.RenderParticles()
        ret = particles.read(self.directory + 'particles%05d.bin' % frame)
        if not ret:
            print 'read file failed'
            return
        image_buffer = tc_core.RGBImageFloat(self.video_manager.width, self.video_manager.height, Vector(0, 0, 0.0))
        camera = Camera('perspective', origin=(0, 0, 50),
                        look_at=(0, 0, 0), up=(0, 1, 0), fov_angle=70,
                        width=self.video_manager.width, height=self.video_manager.height)
        self.particle_renderer.set_camera(camera)
        self.particle_renderer.render(image_buffer, particles)
        img = image_buffer_to_ndarray(image_buffer)
        #img = LDRDisplay(exposure=1, adaptive_exposure=False).process(img)
        cv2.imshow('Vis', img)
        cv2.waitKey(1)
        #self.video_manager.write_frame(img)

    def make_video(self):
        self.video_manager.make_video()

if __name__ == '__main__':
    viewer = ParticleViewer('task-2016-12-10-21-16-05-r09837', 960, 540)
    for i in range(100):
        viewer.view(i * 10)

