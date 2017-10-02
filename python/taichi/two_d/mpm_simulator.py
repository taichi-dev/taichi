from PIL import Image

from .levelset_2d import LevelSet2D
from .simulator import Simulator
from taichi.core import tc_core
from taichi.misc.util import *
from taichi.misc.settings import get_asset_path


class MPMSimulator(Simulator):

  def __init__(self, **kwargs):
    Simulator.__init__(self, kwargs['simulation_time'], kwargs['frame_dt'])
    self.simulator = tc_core.MPMSimulator()
    self.res = kwargs['res']
    kwargs['delta_x'] = 1.0 / min(self.res)
    self.simulator.initialize(config_from_dict(kwargs))
    self.config = kwargs
    self.delta_x = kwargs['delta_x']
    self.sample_rate = kwargs.get('sample_rate', 2)
    self.show_limits = kwargs.get('show_limits', True)

    def dummy_levelset_generator(_):
      return self.create_levelset()

    self.levelset_generator = dummy_levelset_generator

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
    else:
      assert False, 'Unknown particle type:' + str(particle_type)
    return particle

  def modify_particle(self, particle, modifiers, u, v):
    if 'velocity' in modifiers:
      particle.velocity = const_or_evaluate(
          modifiers['velocity'], u, v) / Vector(self.delta_x, self.delta_x)
    if 'compression' in modifiers:
      particle.set_compression(
          const_or_evaluate(modifiers['compression'], u, v))
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
        tc_core.make_range(.25 * self.delta_x, self.res[0] * self.delta_x,
                           self.delta_x / self.sample_rate),
        tc_core.make_range(.25 * self.delta_x, self.res[1] * self.delta_x,
                           self.delta_x / self.sample_rate),
        make_polygon(polygon, 1))
    samples = []
    for p in positions:
      u = p.x
      v = p.y
      particle = self.create_particle(particle_type)
      self.modify_particle(particle, kwargs, u, v)
      particle.position = Vector(p.x / self.delta_x, p.y / self.delta_x)
      samples.append(particle)
    self.add_particles(samples)

  def add_particles_texture(self, center, width, filename, particle_type,
                            **kwargs):
    if filename[0] != '/':
      filename = get_asset_path('texture', filename)
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
      self.modify_particle(particle, kwargs, u, v)
      particle.position = Vector(p.x / self.delta_x, p.y / self.delta_x)
      samples.append(particle)
    self.add_particles(samples)

  def add_particles_sphere(self, center, radius, particle_type, **kwargs):
    positions = tc_core.points_inside_sphere(
        tc_core.make_range(.25 * self.delta_x, self.res[0] * self.delta_x,
                           self.delta_x / self.sample_rate),
        tc_core.make_range(.25 * self.delta_x, self.res[1] * self.delta_x,
                           self.delta_x / self.sample_rate), center, radius)
    samples = []
    for p in positions:
      u = p.x
      v = p.y
      particle = self.create_particle(particle_type)
      self.modify_particle(particle, kwargs, u, v)
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
    t = self.simulator.get_current_time()
    levelset = self.levelset_generator(t)
    images.append(levelset.get_image(width, height, color_scheme['boundary']))
    material_levelset = self.simulator.get_material_levelset()
    images.append(
        array2d_to_image(material_levelset, width, height, color_scheme[
            'material']))
    cover_images = []
    if self.show_limits:
      debug_blocks = self.simulator.get_debug_blocks().rasterize_scale(
          self.res[0], self.res[1], self.simulator.get_grid_block_size())
      debug_blocks = array2d_to_image(
          debug_blocks, width, height, transform=[0, 1], alpha_scale=0.4)
      cover_images.append(debug_blocks)
    return images, cover_images

  def create_levelset(self):
    return LevelSet2D(self.res[0] + 1, self.res[1] + 1, self.delta_x,
                      Vector(0.0, 0.0))

  def test(self):
    return self.simulator.test()
