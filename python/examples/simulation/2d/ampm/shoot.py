from taichi.misc.util import *
from taichi.two_d import *
import taichi as tc

if __name__ == '__main__':
    scale = 64
    res = (scale, scale)
    frame_dt = 1e-2
    async = True
    gravity = (0, 0)
    if async:
        simulator = MPMSimulator(res=res, simulation_time=15, frame_dt=frame_dt, base_delta_t=1e-5, async=True,
                                 maximum_delta_t=1e-2, debug_input=(32, 5, 0, 0), cfl=0.6, gravity=gravity)
    else:
        simulator = MPMSimulator(res=res, simulation_time=15, frame_dt=frame_dt, base_delta_t=1e-4, async=False,
                                 maximum_delta_t=1e-1, debug_input=(1024, 4, 0, 0), gravity=gravity)


    def event(s):
        s.add_particles_sphere(Vector(0.2, 0.5), 0.05, 'ep',
                               compression=0.9, velocity=Vector(50, 0),
                               theta_c=1, theta_s=1)

        center_x, center_y = 0.7, 0.5
        w, h = 0.05, 0.35
        polygon = [
            Vector(center_x - w, center_y - h),
            Vector(center_x + w, center_y - h),
            Vector(center_x + w, center_y + h),
            Vector(center_x - w, center_y + h),
        ]
        s.add_particles_polygon(polygon, 'ep', compression=0.3, velocity=Vector(-0.1, 0))


    simulator.add_event(-1, event)

    levelset = simulator.create_levelset()
    bd = -1
    levelset.add_polygon([(bd, bd), (1 - bd, bd), (1 - bd, 1 - bd), (bd, 1 - bd)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True,
                              substep=async, video_output=False, need_press=False)
