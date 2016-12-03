from vfx import *

if __name__ == '__main__':
    scale = 8
    resolution = tuple([32 * scale, 32 * scale])
    simulator = create_mpm_simulator(resolution, 20, 0.02, 0.001)

    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.7, 0.5), 0.15, 'ep', compression=0.6, theta_s=0.0007,
                                                             velocity=Vector(0.5, 0.0)))

    levelset = simulator.create_levelset()
    levelset.set_friction(0)
    levelset.add_polygon(polygons['square'], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True)
