from taichi.misc.util import *
from taichi.two_d import *

if __name__ == '__main__':
    resolution = tuple([160, 90])
    simulator = create_mpm_simulator(resolution, 20, frame_dt=3e-2, base_delta_t=2e-3)

    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.25), 0.10, 'ep',
                                                             compression=1.1))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.05, 0.05), (1.73, 0.05), (1.73, 0.95), (0.05, .95)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(1280, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True)
