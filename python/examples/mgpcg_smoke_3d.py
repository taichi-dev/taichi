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

class Smoke3:
    def __init__(self, **kwargs):
        self.c = tc_core.create_simulation3d('smoke')
        self.c.initialize(P(**kwargs))
        self.directory = '../output/frames/' + get_uuid() + '/'
        self.video_manager = VideoManager(self.directory, 512, 1024)
        try:
            os.mkdir(self.directory)
        except Exception as e:
            print e
        self.particle_renderer = ParticleRenderer('shadow_map',
                                                  shadow_map_resolution=0.5, alpha=0.2, shadowing=0.07, ambient_light=0.3,
                                                  light_direction=(1, 3, 1))
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
        radius = res[0] * 4
        theta = t * 0.0 + 1.3
        camera = Camera('perspective', origin=(radius * math.cos(theta), radius * 0.3, radius * math.sin(theta)),
                        look_at=(0, 0, 0), up=(0, 1, 0), fov_angle=70,
                        width=self.video_manager.width, height=self.video_manager.height)
        self.particle_renderer.set_camera(camera)
        self.particle_renderer.render(image_buffer, particles)
        img = image_buffer_to_ndarray(image_buffer)
        #img = LDRDisplay(exposure=1, adaptive_exposure=False).process(img)
        cv2.imshow('Vis', img)
        cv2.waitKey(1)
        self.video_manager.write_frame(img)

    def make_video(self):
        self.video_manager.make_video()

if __name__ == '__main__':
    resolution = [128] * 3
    resolution[1] *= 2
    smoke = Smoke3(resolution=tuple(resolution),
                 simulation_depth=resolution[2], delta_x=1.0 / resolution[0], gravity=(0, -10),
                 advection_order=1, cfl=0.5, smoke_alpha=80.0, smoke_beta=800,
                 temperature_decay=0.05, pressure_tolerance=1e-4, density_scaling=2, initial_speed=(0, 0, 0),
                 tracker_generation=20, perturbation=0, pressure_solver='mgpcg', num_threads=8)
    for i in range(600):
        smoke.step(0.03)

    smoke.make_video()


