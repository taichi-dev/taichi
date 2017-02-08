import taichi as tc
import colorsys
from taichi.misc.util import *
from taichi.gui.image_viewer import show_image

if __name__ == '__main__':
    resolution = [64] * 3
    resolution[1] *= 2

    particle_renderer = tc.ParticleRenderer('shadow_map',
                                            shadow_map_resolution=0.5, alpha=0.6, shadowing=0.07, ambient_light=0.3,
                                            light_direction=(1, 3, 1))

    smoke = tc.Smoke3(resolution=tuple(resolution),
                      simulation_depth=resolution[2], delta_x=1.0 / resolution[0], gravity=(0, 0, 0),
                      advection_order=1, cfl=0.5, smoke_alpha=1.0, smoke_beta=1000,
                      temperature_decay=0.05, pressure_tolerance=1e-6, density_scaling=2, initial_speed=(0, 0, 0),
                      tracker_generation=20, perturbation=0, pressure_solver='mgpcg', num_threads=2, open_boundary=True,
                      maximum_pressure_iterations=200, super_sampling=20)

    video_manager = tc.VideoManager(output_dir='new_year')
    images = []
    for i in range(600):
        print 'frame', i
        generation = tc.Texture('sphere', radius=0.08, center=(0.25, 0.1, 0.25)) + \
            tc.Texture('sphere', radius=0.08, center=(0.25, 0.1, 0.75)) + \
            tc.Texture('sphere', radius=0.08, center=(0.75, 0.1, 0.25)) + \
            tc.Texture('sphere', radius=0.08, center=(0.75, 0.1, 0.75))
        smoke.update(generation=generation * 100,
                 color=tc.Texture('const', value=colorsys.hls_to_rgb(0.02 * i + 0.0, 0.7, 1.0)),
                 temperature=tc.Texture('const', value=(1, 0, 0, 0)),
                 initial_velocity=tc.Texture('const', value=(0, 100, 0, 0)))
        smoke.step(0.03)
        particles = smoke.c.get_render_particles()
        width, height = 512, 1024
        image_buffer = tc.core.Array2DVector3(width, height, Vector(0, 0, 0.0))
        radius = resolution[0] * 4
        camera = tc.Camera('pinhole', origin=(0, radius * 0.3, radius),
                        look_at=(0, 0, 0), up=(0, 1, 0), fov=70,
                        width=width, height=height)
        particle_renderer.set_camera(camera)
        particle_renderer.render(image_buffer, particles)
        img = image_buffer_to_ndarray(image_buffer)
        show_image('Vis', img)
        images.append(img)
    video_manager.write_frames(images)
    video_manager.make_video()
