from taichi.two_d import *
from taichi.misc.util import *

if __name__ == '__main__':
  res = tuple([256] * 2)
  simulator = MPMSimulator(
      res=res, simulation_time=10, frame_dt=0.01, base_delta_t=1e-3)

  simulator.add_event(
      -1,
      lambda s: s.add_particles_polygon([(0.45, 0.15), (0.55, 0.15), (0.55, 0.8), (0.45, 0.8)], 'dp')
  )
  simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.23, 0.6), 0.06, 'ep', compression=0.8,
                                                           velocity=Vector(1, 0), color=Vector(255, 255, 255)))
  simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.73, 0.4), 0.06, 'ep',
                                                           theta_c=1, theta_s=1, velocity=Vector(-0.2, -0.5),
                                                           color=Vector(0, 128, 255)))
  for i in range(0):
    t = 5 + i * 2
    f = lambda s: s.add_particles_sphere(Vector(0.5, 0.7), 0.05, 'dp')
    simulator.add_event(t, f)

  levelset = simulator.create_levelset()
  levelset.add_polygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], True)
  simulator.set_levelset(levelset)
  window = SimulationWindow(
      512,
      simulator,
      color_schemes['sand'],
      levelset_supersampling=2,
      show_images=True)
