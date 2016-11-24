from vfx import *

if __name__ == '__main__':
    scale = 8
    resolution = tuple([80 * scale, 40 * scale])
    simulator = create_mpm_simulator(resolution, 30, 0.06, 0.001)

    simulator.add_event(-1, lambda s: s.add_particles_texture(Vector(1, 0.60), 1.8, 'hybrid.png', 'ep', theta_c=1, theta_s=1))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.05, 0.05), (1, 0.4), (1.95, 0.05), (1.95, 0.95), (0.05, 0.95)], True)
    levelset.set_friction(0)
    simulator.set_levelset(levelset)
    window = SimulationWindow(1280, simulator, color_schemes['bw'], levelset_supersampling=2, show_images=True)
