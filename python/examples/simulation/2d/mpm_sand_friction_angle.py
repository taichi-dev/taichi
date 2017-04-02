from taichi.two_d import *
from taichi.misc.util import *

if __name__ == '__main__':
    resolution = tuple([256] * 2)
    simulator = create_mpm_simulator(resolution, 5, frame_dt=0.06)

    simulator.add_event(-1, lambda s: s.add_particles_polygon([(0.45, 0.11), (0.55, 0.11), (0.55, 0.5), (0.45, 0.5)], 'dp', h_0=20))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, simulator, color_schemes['sand'], levelset_supersampling=2, show_images=True)
