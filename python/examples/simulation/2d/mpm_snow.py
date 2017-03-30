from taichi.misc.util import *
from taichi.two_d import *

if __name__ == '__main__':
    resolution = tuple([320, 180])
    simulator = create_mpm_simulator(resolution, 20, frame_dt=3e-2, base_delta_t=1e-3)

    simulator.add_event(-1, lambda s: s.add_particles_polygon([(0.05, 0.75), (1.2, 0.05), (1.2, 0.08), (0.05, 0.78)],
                                                              'ep', compression=1.1))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.12, 0.85), 0.06, 'ep', compression=0.8,
                                                             velocity=Vector(0.0, -0.1)))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.15), 0.10, 'ep', compression=0.8))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.32), 0.07, 'ep', compression=0.8,
                                                             color=Vector(0, 255, 0)))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.43), 0.04, 'ep', compression=0.8))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.05, 0.75), (1.2, 0.05), (1.73, 0.05), (1.73, 0.95), (0.05, .95)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(1280, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True)
