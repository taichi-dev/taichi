from taichi.misc.util import *
from taichi.two_d import *
import taichi as tc

if __name__ == '__main__':
    scale = 64
    res = (scale, scale)
    frame_dt = 1e-3
    async = True
    bullet = True
    gravity = (0, 0)
    if async:
        simulator = MPMSimulator(res=res, simulation_time=0.3, frame_dt=frame_dt, base_delta_t=1e-6, async=True,
                                 maximum_delta_t=2e-1, debug_input=(128, 7, 0, 0), cfl=0.1, gravity=gravity)
    else:
        simulator = MPMSimulator(res=res, simulation_time=0.3, frame_dt=frame_dt, base_delta_t=5e-6, async=False,
                                 maximum_delta_t=1e-1, debug_input=(1024, 4, 0, 0), gravity=gravity)


    def event(s):
        if bullet:
            s.add_particles_sphere(Vector(0.2, 0.5), 0.03, 'ep',
                                   compression=0.1, velocity=Vector(50, 0), lambda_0=1e10, mu_0=1e10,
                                   theta_c=1, theta_s=1)

        center_x, center_y = 0.5, 0.5
        w, h = 0.05, 0.35
        polygon = [
            Vector(center_x - w, center_y - h),
            Vector(center_x + w, center_y - h),
            Vector(center_x + w, center_y + h),
            Vector(center_x - w, center_y + h),
        ]
        s.add_particles_polygon(polygon, 'ep', compression=0.8, velocity=Vector(-0.1, 0))


    simulator.add_event(-1, event)

    levelset = simulator.create_levelset()
    bd = -1
    levelset.add_polygon([(bd, bd), (1 - bd, bd), (1 - bd, 1 - bd), (bd, 1 - bd)], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True,
                              substep=False, video_output=False, need_press=False)
