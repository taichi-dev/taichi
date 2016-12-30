from taichi.vfx import *

if __name__ == '__main__':
    resolution = tuple([64] * 2)
    simulator = create_simulator_sand(resolution, 30)

    for i in range(1000):
        f = lambda s: s.add_particles_sphere(Vector(0.5, 0.9), 0.01, vel_eval=(0.0, -10))
        simulator.add_event(i * 0.06, f)

    levelset = simulator.create_levelset()
    levelset.add_polygon(polygons['T'], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, 512, simulator, color_schemes['sand'], levelset_supersampling=2)
