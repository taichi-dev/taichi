from taichi.dynamics.smoke3 import Smoke3

if __name__ == '__main__':
    resolution = [64] * 3
    resolution[1] *= 2
    smoke = Smoke3(resolution=tuple(resolution),
                 simulation_depth=resolution[2], delta_x=1.0 / resolution[0], gravity=(0, -10),
                 advection_order=1, cfl=0.5, smoke_alpha=80.0, smoke_beta=800,
                 temperature_decay=0.05, pressure_tolerance=1e-4, density_scaling=2, initial_speed=(0, 0, 0),
                 tracker_generation=20, perturbation=0, pressure_solver='mgpcg', num_threads=8)
    for i in range(600):
        smoke.step(0.03)

    smoke.make_video()


