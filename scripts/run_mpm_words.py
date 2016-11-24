from vfx import *

if __name__ == '__main__':
    scale = 10
    resolution = tuple([80 * scale, 45 * scale])
    simulator = create_mpm_simulator(resolution, 30, 0.06, 0.001)

    simulator.add_event(-1, lambda s: s.add_particles_texture(Vector(0.86, 0.30), 1.5, 'aligadou.png', 'ep'))
    simulator.add_event(2, lambda s: s.add_particles_texture(Vector(0.86, 0.45), 1.5, 'lab.png', 'ep', theta_c=1, theta_s=1, color=Vector(48, 122, 120)))
    simulator.add_event(4, lambda s: s.add_particles_texture(Vector(0.86, 0.60), 1.5, 'question.png', 'ep', color=Vector(247, 205, 115)))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.05, 0.05), (1.73, 0.05), (1.73, 0.95), (0.05, 0.95)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(1280, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True)
