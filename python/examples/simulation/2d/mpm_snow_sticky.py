from taichi.two_d import *
from taichi.misc.util import *

if __name__ == '__main__':
    scale = 8
    res = tuple([32 * scale, 32 * scale])
    simulator = MPMSimulator(res=res, simulation_time=20, frame_dt=.02, base_delta_t=1e-3)

    simulator.add_event(-1,
                        lambda s: s.add_particles_sphere(Vector(0.7, 0.5), 0.15, 'ep', compression=0.6, theta_s=0.0007,
                                                         velocity=Vector(0.5, 0.0)))

    levelset = simulator.create_levelset()
    levelset.set_friction(0)
    levelset.add_polygon(polygons['square'], True)
    simulator.set_levelset(levelset)
    window = SimulationWindow(512, simulator, color_schemes['snow'], levelset_supersampling=2, show_images=True)
