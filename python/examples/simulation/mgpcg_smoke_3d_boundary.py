from taichi.util import *
from taichi.core import tc_core
from taichi.dynamics.smoke3 import Smoke3
from taichi.visual.particle_renderer import ParticleRenderer
from taichi.visual.camera import Camera
import cv2

if __name__ == '__main__':
    resolution = [64] * 3
    resolution[1] *= 2

    particle_renderer = ParticleRenderer('shadow_map',
                                         shadow_map_resolution=0.5, alpha=0.6, shadowing=0.07, ambient_light=0.3,
                                         light_direction=(1, 3, 1))

    smoke = Smoke3(resolution=tuple(resolution),
                 simulation_depth=resolution[2], delta_x=1.0 / resolution[0], gravity=(0, 0, 0),
                 advection_order=1, cfl=0.5, smoke_alpha=80.0, smoke_beta=800,
                 temperature_decay=0.05, pressure_tolerance=1e-6, density_scaling=2, initial_speed=(0, 0, 0),
                 tracker_generation=20, perturbation=0, pressure_solver='mgpcg', num_threads=2, open_boundary=True)


    for i in range(600):
        smoke.step(0.03)
        particles = smoke.c.get_render_particles()
        width, height = 512, 1024
        image_buffer = tc_core.RGBImageFloat(width, height, Vector(0, 0, 0.0))
        radius = resolution[0] * 4
        camera = Camera('pinhole', origin=(0, radius * 0.3, radius),
                        look_at=(0, 0, 0), up=(0, 1, 0), fov=70,
                        width=width, height=height)
        particle_renderer.set_camera(camera)
        particle_renderer.render(image_buffer, particles)
        img = image_buffer_to_ndarray(image_buffer)
        cv2.imshow('Vis', img)
        cv2.waitKey(1)


