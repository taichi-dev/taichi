from taichi.misc.util import *
from taichi.two_d import *
import taichi as tc

if __name__ == '__main__':
    scale = 256
    res = (scale, scale)
    frame_dt = 0.003
    async = True
    gravity = (0, 0)
    if async:
        simulator = MPMSimulator(res=res, simulation_time=0.3, frame_dt=frame_dt, base_delta_t=1e-7, async=True,
                                 maximum_delta_t=1e-2, debug_input=(512, 5, 0, 0), cfl=1.0, gravity=gravity)
    else:
        simulator = MPMSimulator(res=res, simulation_time=0.3, frame_dt=frame_dt, base_delta_t=1e-4, async=False,
                                 maximum_delta_t=1e-1, debug_input=(512, 5, 0, 0), gravity=gravity)

    def event(s):
        s.add_particles_sphere(Vector(0.5, 0.5), 0.37, 'ep',
                               compression=1.0, velocity=Vector(0, 0))
        s.add_particles_sphere(Vector(0.5, 0.5), 0.01, 'ep',
                               compression=1.0, velocity=Vector(1, 0), lambda_0=1e10, mu_0=1e10)



    simulator.add_event(-1, event)

    levelset = simulator.create_levelset()
    bd = -1
    levelset.add_polygon([(bd, bd), (1 - bd, bd), (1 - bd, 1 - bd), (bd, 1 - bd)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True,
                              substep=False, video_output=False, need_press=False)
