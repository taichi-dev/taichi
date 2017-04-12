from taichi.misc.util import *
from taichi.two_d import *
import taichi as tc

if __name__ == '__main__':
    res = (160, 90)
    simulator = MPMSimulator(res=res, simulation_time=15, frame_dt=8e-2, base_delta_t=1e-6, async=True, maximum_delta_t=1e-2)

    num_slices = 4


    def get_event(i):
        def event(s):
            print i
            s.add_particles_sphere(Vector(0.3 + 1.2 / (num_slices - 1) * i, 0.20), 0.10, 'ep',
                                   compression=0.6 + i * 0.2)

        return event


    for i in range(num_slices):
        simulator.add_event(-1, get_event(i))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.05, 0.05), (1.73, 0.05), (1.73, 0.95), (0.05, .95)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(1280, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True)
