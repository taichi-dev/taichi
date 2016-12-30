from taichi.mics.util import *


class Simulator(object):
    def __init__(self, simulation_time, dt):
        self.events = []
        self.simulator = None
        self.simulation_time = simulation_time
        self.dt = dt
        self.delta_x = 1
        self.particles = []

    def add_event(self, t, func):
        self.events.append((t, func))
        self.events.sort()

    def step(self):
        t = self.simulator.get_current_time()
        while self.events and t > self.events[0][0]:
            self.events[0][1](self)
            self.events = self.events[1:]
        self.simulator.step(self.dt)
        try:
            self.particles = self.simulator.get_particles()
        except:
            self.particles = []

    def __getattr__(self, key):
        return self.simulator.__getattribute__(key)

    def add_particles(self, particles):
        for p in particles:
            self.add_particle(p)

    def add_source(self, **kwargs):
        self.simulator.add_source(config_from_dict(self.maginify_config(kwargs, ['center', 'radius'])))

    def ended(self):
        return self.simulator.get_current_time() >= self.simulation_time

    def set_levelset(self, levelset):
        self.simulator.set_levelset(levelset.levelset)
        self.levelset = levelset

    def get_levelset_images(self, width, height, color_scheme):
        images = []
        images.append(self.levelset.get_image(width, height, color_scheme['boundary']))
        return images

    def maginify(self, val):
        if type(val) in [int, tc.Vector2, float]:
            return val / self.delta_x
        elif type(val) == list:
            return list(map(lambda x: x / self.delta_x, val))
        elif type(val) == tuple:
            return tuple(map(lambda x: x / self.delta_x, val))
        else:
            assert False

    def maginify_config(self, cfg, keys):
        cfg = copy.copy(cfg)
        for k in keys:
            cfg[k] = self.maginify(cfg[k])
        return cfg


    @staticmethod
    def config_from_dict(dict):
        d = copy.deepcopy(dict)
        for k in d:
            d[k] = str(d[k])
        return tc.config_from_dict(d)

    def get_background_image(self, width, height):
        return None
