from taichi.misc.util import *
from taichi.two_d import *
import taichi as tc

if __name__ == '__main__':
    res = (64, 64)
    simulator = MPMSimulator(res=res, simulation_time=15, frame_dt=1e-2, base_delta_t=1e-5, async=True,
                             maximum_delta_t=1e-1, debug_input=(1, 10, 0, 0), cfl=0.02, gravity=(0, 0))
    # simulator = MPMSimulator(res=res, simulation_time=15, frame_dt=1e-2, base_delta_t=1e-3, async=False,
    #                         maximum_delta_t=1e-1, debug_input=(1024, 4, 0, 0))

    num_slices = 4


    def get_event(i):
        def event(s):
            print i
            s.add_particles_sphere(Vector(0.7 + 1.2 / (num_slices - 1) * i, 0.40), 0.10, 'ep',
                                   compression=1.5 - i * 0.0, velocity=Vector(0, -1))

        return event


    # for i in range(num_slices):
    for i in range(1):
        simulator.add_event(-1, get_event(i))

    levelset = simulator.create_levelset()
    levelset.add_polygon([(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, .95)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True)
