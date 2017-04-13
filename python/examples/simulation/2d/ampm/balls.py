from taichi.misc.util import *
from taichi.two_d import *
import taichi as tc

if __name__ == '__main__':
    scale = 32
    res = (scale, scale)
    frame_dt = 1e-2
    async = False
    gravity = (0, 0)
    if async:
        simulator = MPMSimulator(res=res, simulation_time=15, frame_dt=frame_dt, base_delta_t=1e-5, async=True,
                                 maximum_delta_t=1e-2, debug_input=(32, 10, 0, 0), cfl=0.5, gravity=gravity)
    else:
        simulator = MPMSimulator(res=res, simulation_time=15, frame_dt=frame_dt, base_delta_t=1e-3, async=False,
                                 maximum_delta_t=1e-1, debug_input=(1024, 4, 0, 0), gravity=gravity)

    num_slices = 4


    def get_event(i):
        def event(s):
            print i
            # s.add_particles_sphere(Vector(0.5, 0.5), 0.3, 'ep',
            #                       compression=1, velocity=lambda x, y: Vector(-(y - 0.5), x - 0.5),
            #                       theta_c=1, theta_s=1)
            w, h = 0.05, 0.35
            polygon = [
                Vector(0.5 - w, 0.5 - h),
                Vector(0.5 + w, 0.5 - h),
                Vector(0.5 + w, 0.5 + h),
                Vector(0.5 - w, 0.5 + h),
            ]
            s.add_particles_polygon(polygon, 'ep',
                                    compression=1.0, velocity=lambda x, y: 10 * Vector(-(y - 0.5), x - 0.5),
                                    theta_c=1, theta_s=1)

        return event


    for i in range(1):
        simulator.add_event(-1, get_event(i))

    levelset = simulator.create_levelset()
    bd = -1
    levelset.add_polygon([(bd, bd), (1 - bd, bd), (1 - bd, 1 - bd), (bd, 1 - bd)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True,
                              substep=False, video_output=False)
