from taichi.misc.util import *
from taichi.two_d import *

if __name__ == '__main__':
    scale = 4
    res = (80 * scale, 45 * scale)
    sim_t = 20
    async = False
    gravity = (0, -20)

    if async:
        simulator = MPMSimulator(res=res, simulation_time=sim_t, frame_dt=0.3, base_delta_t=1e-6, async=async,
                                 gravity=gravity, debug_input=(128, 7, 1, 0), strength_dt_mul=10)
    else:
        # dt = 2e-3 is the allowed maximum. DO NOT MODIFY.
        simulator = MPMSimulator(res=res, simulation_time=sim_t, frame_dt=0.3, base_delta_t=1e-3, gravity=gravity)

    simulator.add_event(-1, lambda s: s.add_particles_polygon([(0.05, 0.75), (1.2, 0.05), (1.3, 0.05), (0.05, 0.80)],
                                                              'ep', compression=1.3))
    simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(0.12, 0.85), 0.05, 'ep', compression=0.8,
                                                             velocity=Vector(0.1, -0.2)))

    # simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.15), 0.10, 'ep', compression=0.9))
    # simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.32), 0.07, 'ep', compression=0.9))
    # simulator.add_event(-1, lambda s: s.add_particles_sphere(Vector(1.45, 0.43), 0.04, 'ep', compression=0.9))

    levelset = simulator.create_levelset()
    levelset.set_friction(10)
    levelset.add_polygon([(0.05, 0.75), (1.2, 0.05), (1.73, 0.05), (1.73, 0.95), (0.05, .95)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(640, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True,
                              substep=False, video_output=False)
