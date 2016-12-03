from vfx import *

class SmokeSimulator3D(FluidSimulator):
    def __init__(self, **kwargs):
        super(SmokeSimulator3D, self).__init__(**kwargs)

    def get_background_image(self, width, height):
        return image_buffer_to_image(self.get_visualization(width, height))

    def get_levelset_images(self, width, height, color_scheme):
        return []

    def get_particles(self):
        return []

if __name__ == '__main__':
    resolution = [64] * 3
    resolution[1] *= 2
    simulator = SmokeSimulator3D(simulator='Smoke3D', simulation_width=resolution[0], simulation_height=resolution[1],
                                 simulation_depth=resolution[2], delta_x=1.0 / resolution[0], gravity=(0, -10),
                                 advection_order=1, cfl=0.5, simulation_time=30, dt=0.1, smoke_alpha=80.0, smoke_beta=200,
                                 temperature_decay=0.05, pressure_tolerance=1e-4, density_scaling=2,
                                 initial_speed=(0, 10, 0),
                                 tracker_generation=20, shadow_map_resolution=2048, shadowing=0.03, perturbation=0,
                                 light_direction=(1, 1, 1), viewport_rotation=0.1, pressure_solver='mgpcg')

    window = SimulationWindow(800, simulator, color_schemes['smoke'], rescale=True)

