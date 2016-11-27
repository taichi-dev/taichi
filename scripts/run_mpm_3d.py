from vfx import *

from taichi_utils import *
import random

class MPMSimulator3D(Simulator):
    def __init__(self, **kwargs):
        Simulator.__init__(self, kwargs['simulation_time'], kwargs['dt'])
        self.simulator = tc.MPM3D()
        self.resolution = (kwargs['simulation_width'], kwargs['simulation_height'])
        self.simulation_width, self.simulation_height = self.resolution[0], self.resolution[1]
        self.simulator.initialize(config_from_dict(kwargs))
        self.config = kwargs
        self.delta_x = kwargs['delta_x']
        self.sample_rate = kwargs.get('sample_rate', 2)

    def get_background_image(self, width, height):
        return image_buffer_to_image(self.get_visualization(width, height))

    def get_levelset_images(self, width, height, color_scheme):
        return []

    def get_particles(self):
        return []

if __name__ == '__main__':
    resolution = [16, 16, 16]
    simulator = MPMSimulator3D(simulator='MPM_3D', simulation_width=resolution[0], simulation_height=resolution[1],
                                 simulation_depth=resolution[2], delta_x=1.0 / resolution[0], gravity=(-10, 0, 0),
                                 advection_order=1, cfl=0.5, simulation_time=40, dt=0.03, substeps=10,
                                 shadow_map_resolution=64, shadowing=0.5,
                                 light_direction=(1, 1, 1), viewport_rotation=0.5,
                                 initial_velocity=(0, 0, 0))

    window = SimulationWindow(400, simulator, color_schemes['smoke'], rescale=True)

