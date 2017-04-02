from PIL import Image

from levelset_2d import LevelSet2D
from simulator import Simulator
from taichi.core import tc_core
from taichi.misc.util import *


class MPMSimulator(Simulator):
    def __init__(self, **kwargs):
        Simulator.__init__(self, kwargs['simulation_time'], kwargs['dt'])
        self.simulator = tc_core.MPMSimulator()
        self.resolution = kwargs['res']
        self.simulation_width, self.simulation_height = self.resolution[0], self.resolution[1]
        self.simulator.initialize(config_from_dict(kwargs))
        self.config = kwargs
        self.delta_x = kwargs['delta_x']
        self.sample_rate = kwargs.get('sample_rate', 2)

    @staticmethod
    def create_particle(particle_type):
        if particle_type == 'ep':
            particle = tc_core.EPParticle()
            particle.mu_0 = 1e6
            particle.lambda_0 = 2e5
            particle.theta_c = 0.01
            particle.theta_s = 0.005
            particle.hardening = 5
        elif particle_type == 'dp':
            particle = tc_core.DPParticle()
            particle.mu_0 = 1e6
            particle.lambda_0 = 2e5
            particle.mu_0 = 10000000
            particle.lambda_0 = 10000000
            particle.h_0 = 45
            particle.h_1 = 9
            particle.h_2 = 0.2
            particle.h_3 = 10
            particle.alpha = 1
        return particle

    def add_particles_rect(self, x, y, delta_x=0, vel_eval=None, comp_eval=None):
        x_0, x_1 = x[0], x[1]
        y_0, y_1 = y[0], y[1]
        if delta_x <= 0.0:
            delta_x = self.delta_x / 2
        samples = []
        x = x_0
        while x < x_1:
            y = y_0
            while y < y_1:
                y += delta_x
                u = (x - x_0) / (x_1 - x_0)
                v = (y - y_0) / (y_1 - y_0)
                vel = default_const_or_evaluate(vel_eval, (0, 0), u, v)
                comp = default_const_or_evaluate(comp_eval, 1.0, u, v)
                samples.append(
                    MPMParticle(Vector(x / self.delta_x, y / self.delta_x), Vector(vel[0], vel[1]), comp, -1))
            x += delta_x
        self.add_particles(samples)

    def modifty_particle(self, particle, modifiers, u, v):
        if 'velocity' in modifiers:
            particle.velocity = const_or_evaluate(modifiers['velocity'], u, v) / Vector(self.delta_x, self.delta_x)
        if 'compression' in modifiers:
            particle.set_compression(const_or_evaluate(modifiers['compression'], u, v))
        if 'color' in modifiers:
            particle.color = const_or_evaluate(modifiers['color'], u, v)
        if 'theta_c' in modifiers:
            particle.theta_c = const_or_evaluate(modifiers['theta_c'], u, v)
        if 'theta_s' in modifiers:
            particle.theta_s = const_or_evaluate(modifiers['theta_s'], u, v)
        if 'lambda_0' in modifiers:
            particle.lambda_0 = const_or_evaluate(modifiers['lambda_0'], u, v)
        if 'mu_0' in modifiers:
            particle.lambda_0 = const_or_evaluate(modifiers['mu_0'], u, v)
        if 'h_0' in modifiers:
            particle.h_0 = const_or_evaluate(modifiers['h_0'], u, v)

    def add_particles_polygon(self, polygon, particle_type, **kwargs):
        positions = tc_core.points_inside_polygon(
            tc_core.make_range(.25 * self.delta_x, self.resolution[0] * self.delta_x, self.delta_x / self.sample_rate),
            tc_core.make_range(.25 * self.delta_x, self.resolution[1] * self.delta_x, self.delta_x / self.sample_rate),
            make_polygon(polygon, 1)
        )
        samples = []
        for p in positions:
            u = p.x
            v = p.y
            particle = self.create_particle(particle_type)
            self.modifty_particle(particle, kwargs, u, v)
            particle.position = Vector(p.x / self.delta_x, p.y / self.delta_x)
            samples.append(particle)
        self.add_particles(samples)

    def add_particles_texture(self, center, width, filename, particle_type, **kwargs):
        if filename[0] != '/':
            filename = TEXTURE_PATH + filename
        im = Image.open(filename)
        positions = []
        height = width / im.width * im.height
        rwidth = int(width / self.delta_x * 2)
        rheight = int(height / self.delta_x * 2)
        im = im.resize((rwidth, rheight))
        rgb_im = im.convert('RGB')
        for i in range(rwidth):
            for j in range(rheight):
                x = center.x + 0.5 * i * self.delta_x - width / 2
                y = center.y + 0.5 * j * self.delta_x - height / 2
                r = 1 - rgb_im.getpixel((i, rheight - 1 - j))[0] / 255.
                if random.random() < r:
                    positions.append(Vector(x, y))

        samples = []
        for p in positions:
            u = p.x
            v = p.y
            particle = self.create_particle(particle_type)
            self.modifty_particle(particle, kwargs, u, v)
            particle.position = Vector(p.x / self.delta_x, p.y / self.delta_x)
            samples.append(particle)
        self.add_particles(samples)

    def add_particles_sphere(self, center, radius, particle_type, **kwargs):
        positions = tc_core.points_inside_sphere(
            tc_core.make_range(.25 * self.delta_x, self.resolution[0] * self.delta_x, self.delta_x / self.sample_rate),
            tc_core.make_range(.25 * self.delta_x, self.resolution[1] * self.delta_x, self.delta_x / self.sample_rate),
            center, radius
        )
        samples = []
        for p in positions:
            u = p.x
            v = p.y
            particle = self.create_particle(particle_type)
            self.modifty_particle(particle, kwargs, u, v)
            particle.position = Vector(p.x / self.delta_x, p.y / self.delta_x)
            samples.append(particle)
        self.add_particles(samples)

    def add_particles(self, particles):
        for p in particles:
            if isinstance(p, tc_core.EPParticle):
                self.add_ep_particle(p)
            elif isinstance(p, tc_core.DPParticle):
                self.add_dp_particle(p)

    def get_levelset_images(self, width, height, color_scheme):
        images = []
        images.append(self.levelset.get_image(width, height, color_scheme['boundary']))
        material_levelset = self.get_material_levelset()
        images.append(array2d_to_image(material_levelset, width, height, color_scheme['material']))
        return images

    def create_levelset(self):
        return LevelSet2D(self.simulation_width, self.simulation_height,
                          self.delta_x, Vector(0.5, 0.5))


def create_mpm_simulator(resolution, t, frame_dt, base_delta_t=1e-3, dt_multiplier=None, **kwargs):
    return MPMSimulator(res=resolution,
                        delta_x=1.0 / min(resolution),
                        gravity=(0, -20),
                        position_noise=0.5,
                        use_level_set=True,
                        particle_collision=True,
                        apic=True,
                        implicit_ratio=0.0,
                        base_delta_t=base_delta_t,
                        maximum_iterations=200,
                        threads=1,
                        flip_alpha=0.0,
                        flip_alpha_stride=1.0,
                        cfl=0.5,
                        simulation_time=t,
                        dt=frame_dt,
                        sample_rate=2,
                        dt_multiplier_tex_id=dt_multiplier.id if dt_multiplier else -1,
                        **kwargs
                        )
