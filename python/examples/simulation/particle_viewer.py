import math

import cv2

from taichi.core import tc_core
from taichi.misc.util import *
from taichi.tools.video import VideoManager
from taichi.visual.camera import Camera
from taichi.visual.particle_renderer import ParticleRenderer


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

    def view(self, frame, camera):
        particles = tc_core.RenderParticles()
        ret = particles.read(self.input_directory + 'particles%05d.bin' % frame)
        if not ret:
            print 'read file failed'
            return False
        image_buffer = tc_core.RGBImageFloat(self.video_manager.width, self.video_manager.height, Vector(0, 0, 0.0))
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

def get_camera(t):
    t = 0.5 * (1 - math.cos(math.pi * t))
    radius = 350
    origin = (math.sin(t * 2 * math.pi) * radius, 25, math.cos(t * 2 * math.pi) * radius)
    camera = Camera('pinhole', origin=origin,
                    look_at=(0, 0, 0), up=(0, 1, 0), fov=90,
                    width=width, height=height)
    return camera

if __name__ == '__main__':
    width, height = 960, 540
    viewer = ParticleViewer('snow-taichi-g10', width, height)
    radius = 350
    framerate = 3
    rotate_start = 30
    rotate_period = 120
    for frame in range(420):
        if frame < rotate_start or frame >= rotate_start + rotate_period:
            dat = frame * framerate
            if frame >= rotate_start + rotate_period:
                dat -= rotate_period * framerate
            camera = get_camera(0)
        else:
            # Rotate view
            camera = get_camera(1.0 * (frame - rotate_start) / rotate_period)
            dat = rotate_start * framerate
        ret = viewer.view(dat, camera)
    viewer.make_video()
