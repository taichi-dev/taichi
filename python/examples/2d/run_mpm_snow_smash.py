from taichi.two_d import *
from taichi.misc.util import *

if __name__ == '__main__':
    scale = 4
    resolution = tuple([80 * scale, 45 * scale])
    simulator = create_mpm_simulator(resolution, 20, 0.02, 0.001)

    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.32, 0.5), 0.10, 'ep', compression=0.8,
                                                             velocity=Vector(1.0, 0.0), theta_s=0.002))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.32, 0.55), 0.10, 'ep', compression=0.8,
                                                             velocity=Vector(-1.0, 0.0), theta_s=0.002))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.05, 0.05), (1.73, 0.05), (1.73, 0.95), (0.05, 0.95)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(1280, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True)
