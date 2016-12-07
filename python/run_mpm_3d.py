from taichi.util import *
from taichi.core import tc_core
from taichi.tools.video import VideoManager
import random
import time
import cv2

class MPM3:
    def __init__(self, **kwargs):
        self.c = tc_core.MPM3D()
        self.c.initialize(P(**kwargs))
        self.video_manager = VideoManager(get_uuid(), 512, 512)

    def step(self, t):
        print self.c.get_current_time()
        T = time.time()
        self.c.step(t)
        print 'Time:', time.time() - T
        img = image_buffer_to_ndarray(self.c.get_visualization(self.video_manager.width, self.video_manager.height))
        cv2.imshow('Vis', img)
        cv2.waitKey(1)
        self.video_manager.write_frame(img)

    def make_video(self):
        self.video_manager.make_video()


if __name__ == '__main__':
    resolution = [128] * 3
    mpm = MPM3(simulation_width=resolution[0], simulation_height=resolution[1], simulation_depth=resolution[2],
               gravity=(0, -10, 0), initial_velocity=(0, -10, 0), delta_t=0.002, shadow_map_resolution=64,
               shadowing=0.5, light_direction=(1, 1, 1), num_threads=8)
    for i in range(100):
        mpm.step(0.05)

    mpm.make_video()

