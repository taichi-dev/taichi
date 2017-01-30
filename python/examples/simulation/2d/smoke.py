from taichi.two_d import *
from taichi.misc.util import *

if __name__ == '__main__':
    resolution = [64] * 2
    simulator = SmokeSimulator(simulator='EulerSmoke', simulation_width=resolution[0],
                               simulation_height=resolution[1],
                               delta_x=1.0 / resolution[0], gravity=(0, -10),
                               initialize_particles=False, correction_strength=0.5,
                               correction_neighbours=5, advection_order=1,
                               use_bridson_pcg=False, flip_alpha=0.95, padding=0.05, resizable=True, cfl=0.5,
                               simulation_time=60, dt=0.1, buoyancy_alpha=0.1, buoyancy_beta=1.5)

    levelset = simulator.create_levelset()
    levelset.add_polygon(polygons['maze'], True)
    simulator.set_levelset(levelset)
    simulator.add_source(center=(0.5, 0.2), radius=0.08, emission=2000, density=1, velocity=(0, 0), temperature=20)
    window = SimulationWindow(512, simulator, color_schemes['smoke'], show_grid=False)
