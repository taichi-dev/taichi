import time

import taichi
from taichi.core import tc_core
from taichi.misc.util import *


class Smoke3:
    def __init__(self, **kwargs):
        self.c = tc_core.create_simulation3d('smoke')
        self.c.initialize(P(**kwargs))
        self.directory = taichi.get_output_path(get_unique_task_id())
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

    def update(self, generation, initial_velocity, color, temperature):
        cfg = P(
            generation_tex=generation.id,
            initial_velocity_tex=initial_velocity.id,
            color_tex=color.id,
            temperature_tex=temperature.id,
        )
        self.c.update(cfg)
