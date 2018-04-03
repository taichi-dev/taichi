from taichi.two_d import *
from taichi.misc.util import *

if __name__ == '__main__':
  resolution = [96, 54]
  simulator = FluidSimulator(
      simulator='liquid',
      simulation_width=resolution[0],
      simulation_height=resolution[1],
      delta_x=1.0 / min(resolution),
      gravity=(0, -10),
      initialize_particles=False,
      correction_strength=0.0,
      correction_neighbours=5,
      advection_order=1,
      flip_alpha=0.95,
      padding=0.05,
      cfl=0.5,
      simulation_time=50,
      dt=0.1,
      levelset_band=3)

  simulator.add_event(-1, lambda simulator: simulator.add_particles_rect(x=(0.1, 0.6), y=(0.05, 0.7)))

  levelset = simulator.create_levelset()
  levelset.add_polygon([(0.05, 0.05), (1.0, 0.05), (1.2, 0.2), (1.73, 0.05),
                        (1.73, 0.95), (0.78, 0.95), (0.58, 0.8), (0.05, 0.95)],
                       True)
  simulator.set_levelset(levelset)
  window = SimulationWindow(
      640, simulator, color_schemes['liquid'], levelset_supersampling=2)
