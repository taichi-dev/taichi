from taichi.two_d import *
from taichi.misc.util import *

if __name__ == '__main__':
  resolution = [128] * 2
  simulator = FluidSimulator(
      simulator='apic_liquid',
      simulation_width=resolution[0],
      simulation_height=resolution[1],
      delta_x=1.0 / resolution[0],
      gravity=(0, -10),
      initialize_particles=False,
      correction_strength=0.0,
      correction_neighbours=5,
      advection_order=1,
      flip_alpha=0.95,
      padding=0.05,
      resizable=True,
      cfl=0.1,
      simulation_time=50,
      dt=0.1)

  for i in range(0, 1000):
    f = lambda s: s.add_particles_sphere(Vector(0.8, 0.8), 0.05, vel_eval=(-10.0, 0))
    simulator.add_event(i * 3.5 - 0.01, f)
    f = lambda s: s.add_particles_sphere(Vector(0.2, 0.75), 0.05, vel_eval=(10.0, 0))
    simulator.add_event(i * 3.5 - 0.01, f)

  levelset = simulator.create_levelset()
  levelset.add_polygon(polygons['T'], True)
  simulator.set_levelset(levelset)
  window = SimulationWindow(
      512, simulator, color_schemes['liquid'], levelset_supersampling=2)
