import math
import taichi as tc
from taichi.two_d import *
from taichi.misc.util import *

if __name__ == '__main__':
    resolution = tuple([256] * 2)
    simulator = create_mpm_simulator(resolution, 1, frame_dt=0.05)

    simulator.add_event(-1, lambda s: s.add_particles_polygon([(0.45, 0.15), (0.55, 0.15), (0.55, 0.8), (0.45, 0.8)], 'dp', h_0=20))

    # Static Levelset
    # levelset = simulator.create_levelset()
    # levelset.add_polygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], True)
    # simulator.set_levelset(levelset)

    # Dynamic Levelset
    def levelset_generator(t):
        levelset = simulator.create_levelset()
        velocity = (0.5 + 0.25 * t) * 3.1415
        levelset.add_sphere(Vector(0.5 + 0.25 * math.cos(t * velocity), 0.5 + 0.25 * math.sin(t * velocity)), 0.1, False)
        levelset.add_sphere(Vector(0.5 + 0.25 * math.cos(t * velocity + math.pi), 0.5 + 0.25 * math.sin(t * velocity + math.pi)), 0.1, False)
        levelset.add_sphere(Vector(0.5, 0.5), 0.45, True)
        return levelset
    simulator.set_levelset(levelset_generator, True)

    window = SimulationWindow(512, simulator, color_schemes['sand'], levelset_supersampling=2, show_images=True)
