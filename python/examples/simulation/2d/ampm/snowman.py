from taichi.misc.util import *
from taichi.two_d import *

if __name__ == '__main__':
    res = (320, 180)
    sim_t = 20
    async = False

    if async:
        simulator = MPMSimulator(res=res, simulation_time=sim_t, frame_dt=0.3, base_delta_t=1e-6, async=async,
                                 debug_input=(128, 7, 1, 0), strength_dt_mul=20)
    else:
        simulator = MPMSimulator(res=res, simulation_time=sim_t, frame_dt=0.3, base_delta_t=2e-3)

    simulator.add_event(-1, lambda s: s.add_particles_polygon([(0.05, 0.75), (1.2, 0.05), (1.2, 0.08), (0.05, 0.78)],
                                                              'ep', compression=1.1))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.12, 0.85), 0.05, 'ep', compression=0.8,
                                                             velocity=Vector(0.0, -0.1)))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.15), 0.10, 'ep', compression=0.9))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.32), 0.07, 'ep', compression=0.9))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.43), 0.04, 'ep', compression=0.9))

    levelset = simulator.create_levelset()
    levelset.set_friction(10)
    levelset.add_polygon([(0.05, 0.75), (1.2, 0.05), (1.73, 0.05), (1.73, 0.95), (0.05, .95)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(640, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True, video_output=False)
