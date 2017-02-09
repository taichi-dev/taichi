from taichi.two_d import *
from taichi.misc.util import *

if __name__ == '__main__':
    resolution = [128, 128]
    simulator = FluidSimulator(simulator='liquid', simulation_width=resolution[0],
                               simulation_height=resolution[1],
                               delta_x=1.0 / min(resolution), gravity=(0, -10),
                               initialize_particles=False, correction_strength=0.0,
                               correction_neighbours=5, advection_order=1, flip_alpha=0.95, padding=0.05, cfl=0.5,
                               simulation_time=50, dt=0.1, levelset_band=3, tolerance=1e-4, maximum_iterations=1000)

    simulator.add_event(-1, lambda simulator: simulator.add_particles_rect(x=(0.25, 0.75), y=(0.25, 0.55)))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)], True)
    # levelset.add_polygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(320, simulator, color_schemes['liquid'], levelset_supersampling=2)
