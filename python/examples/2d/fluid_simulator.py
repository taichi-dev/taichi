from levelset import LevelSet2D
from simulator import Simulator
from taichi.mics.util import *


class FluidSimulator(Simulator):
    def __init__(self, **kwargs):
        Simulator.__init__(self, kwargs['simulation_time'], kwargs['dt'])
        simulator_name = kwargs['simulator']
        self.simulator = tc.__dict__[simulator_name]()
        self.simulator.initialize(Simulator.config_from_dict(kwargs))
        self.config = kwargs
        self.delta_x = kwargs['delta_x']
        self.resolution = [kwargs['simulation_width'], kwargs['simulation_height']]
        self.simulation_width, self.simulation_height = self.resolution[0], self.resolution[1]
        self.sample_rate = kwargs.get('sample_rate', 2)
        self.show_pressure = kwargs.get('show_pressure', False)

    def add_particles_rect(self, x, y, delta_x=0, vel_eval=None):
        x_0, x_1 = x[0], x[1]
        y_0, y_1 = y[0], y[1]
        if delta_x <= 0.0:
            delta_x = self.delta_x / self.sample_rate
        samples = []
        x = x_0
        while x < x_1:
            y = y_0
            while y < y_1:
                y += delta_x
                if vel_eval:
                    vel = vel_eval(x - x_0, y - y_0)
                else:
                    vel = (0, 0)
                samples.append(tc.FluidParticle(Vector(x / self.delta_x, y / self.delta_x), Vector(vel[0], vel[1])))
            x += delta_x
        self.add_particles(samples)

    def add_particles_sphere(self, center, radius, vel_eval=None):
        positions = tc.points_inside_sphere(
            tc.make_range(.25 * self.delta_x, self.resolution[0] * self.delta_x,
                             self.delta_x / self.sample_rate),
            tc.make_range(.25 * self.delta_x, self.resolution[1] * self.delta_x,
                             self.delta_x / self.sample_rate),
            center, radius
        )
        samples = []
        for p in positions:
            u = p.x
            v = p.y
            vel = default_const_or_evaluate(vel_eval, (0, 0), u, v)
            samples.append(
                tc.FluidParticle(Vector(p.x / self.delta_x, p.y / self.delta_x), Vector(vel[0], vel[1])))
        self.add_particles(samples)

    def get_levelset_images(self, width, height, color_scheme):
        images = []
        images.append(self.levelset.get_image(width, height, color_scheme['boundary']))
        liquid_levelset = self.get_liquid_levelset()
        images.append(array2d_to_image(liquid_levelset, width, height, color_scheme['liquid']))
        if self.show_pressure:
            images.append(array2d_to_image(self.get_pressure(), width, height, (255, 0, 0, 128), (0, 1)))
        return images

    def create_levelset(self):
        return LevelSet2D(self.simulation_width + 1, self.simulation_height + 1,
                          self.delta_x, Vector(0.0, 0.0))


class SmokeSimulator(FluidSimulator):
    def __init__(self, **kwargs):
        super(SmokeSimulator, self).__init__(**kwargs)

    def get_levelset_images(self, width, height, color_scheme):
        images = []
        images.append(self.levelset.get_image(width, height, color_scheme['boundary']))
        images.append(array2d_to_image(self.get_density(), width, height, color_scheme['smoke'], (0, 5)))
        return images

