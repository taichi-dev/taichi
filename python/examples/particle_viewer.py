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
        self.directory = '../output/frames/' + get_uuid() + '/'
        self.input_directory = '../output/frames/' + directory + '/'
        self.video_manager = VideoManager(self.directory, width, height)
        self.particle_renderer = ParticleRenderer('shadow_map',
                                                  shadow_map_resolution=0.5, alpha=0.5, shadowing=0.1, ambient_light=0.2,
                                                  light_direction=(-1, 1, -1))
        try:
            os.mkdir(self.directory)
        except Exception as e:
            print e
        self.step_counter = 0

    def view(self, frame):
        particles = tc_core.RenderParticles()
        ret = particles.read(self.input_directory + 'particles%05d.bin' % frame)
        if not ret:
            print 'read file failed'
            return False
        image_buffer = tc_core.RGBImageFloat(self.video_manager.width, self.video_manager.height, Vector(0, 0, 0.0))
        camera = Camera('perspective', origin=(0, 50, 350),
                        look_at=(0, 0, 0), up=(0, 1, 0), fov_angle=90,
                        width=self.video_manager.width, height=self.video_manager.height)
        self.particle_renderer.set_camera(camera)
        self.particle_renderer.render(image_buffer, particles)
        img = image_buffer_to_ndarray(image_buffer)
        #img = LDRDisplay(exposure=1, adaptive_exposure=False).process(img)
        cv2.imshow('Vis', img)
        cv2.waitKey(1)
        self.video_manager.write_frame(img)
        return True

    def make_video(self):
        self.video_manager.make_video()

if __name__ == '__main__':
    viewer = ParticleViewer('task-2016-12-10-22-34-13-r03331', 960, 540)
    frame = 0
    while True:
        print frame
        ret = viewer.view(frame)
        frame += 2
        if not ret:
            break
    viewer.make_video()
