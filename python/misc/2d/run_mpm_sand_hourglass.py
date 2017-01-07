from taichi.vfx import *

if __name__ == '__main__':

    resolution = tuple([64] * 2)
    simulator = create_simulator_sand(resolution, 20)

    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.5, 0.7), 0.15))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.0, 0.0), (1, 0), (0.57, 0.5), (1, 1), (0, 1), (0.43, 0.5)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, 512, simulator, color_schemes['sand'], levelset_supersampling=2)
