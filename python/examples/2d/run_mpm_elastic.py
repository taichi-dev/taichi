from taichi.vfx import *

if __name__ == '__main__':
    resolution = tuple([128, 256])
    simulator = create_mpm_simulator(resolution, 10, dt=0.06)

    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.2, 0.55), 0.03, 'ep',
                                                             lambda_0=1e6, mu_0=1e6, theta_c=1, theta_s=1, velocity=Vector(0, 0), color=Vector(255, 255, 255)))

    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.45, 0.4), 0.1, 'ep',
                                                             lambda_0=1e6, mu_0=1e6, theta_c=1, theta_s=1, velocity=Vector(0, 0), color=Vector(255, 255, 255)))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.45, 0.55), 0.03, 'ep',
                                                             lambda_0=1e6, mu_0=1e6, theta_c=1, theta_s=1, velocity=Vector(0, 0), color=Vector(255, 255, 255)))


    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.75, 0.28), 0.1, 'ep',
                                                             lambda_0=1e6, mu_0=1e6, theta_c=1, theta_s=1, velocity=Vector(0, 0), color=Vector(255, 255, 255)))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.75, 0.45), 0.05, 'ep',
                                                             lambda_0=1e6, mu_0=1e6, theta_c=1, theta_s=1, velocity=Vector(0, 0), color=Vector(255, 255, 255)))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.75, 0.55), 0.03, 'ep',
                                                             lambda_0=1e6, mu_0=1e6, theta_c=1, theta_s=1, velocity=Vector(0, 0), color=Vector(255, 255, 255)))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.05, 0.05), (0.95, 0.05), (0.95, 1.95), (0.05, 1.95)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, simulator, color_schemes['sand'], levelset_supersampling=2, show_images=True)
