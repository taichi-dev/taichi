from taichi.util import *
from taichi.core import tc_core
from taichi.visual.camera import Camera
import time
import math

class Smoke3:
    def __init__(self, **kwargs):
        self.c = tc_core.create_simulation3d('smoke')
        self.c.initialize(P(**kwargs))
        self.directory = '../output/frames/' + get_uuid() + '/'
        try:
            os.mkdir(self.directory)
        except Exception as e:
            print e

    def step(self, step_t):
        t = self.c.get_current_time()
        print 'Simulation time:', t
        T = time.time()
        self.c.step(step_t)
        print 'Time:', time.time() - T
