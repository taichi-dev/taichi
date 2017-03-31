import time

from taichi.core import tc_core
from taichi.misc.util import *
from taichi.tools.video import VideoManager
from taichi.visual.camera import Camera
from taichi.visual.particle_renderer import ParticleRenderer
from taichi.visual.post_process import LDRDisplay
from taichi.gui.image_viewer import show_image
import taichi as tc


class MPM3:
    def __init__(self, **kwargs):
        self.c = tc_core.create_simulation3d('mpm')
        self.c.initialize(P(**kwargs))
        self.task_id = get_unique_task_id()
        self.directory = tc.get_output_path(self.task_id)
        self.video_manager = VideoManager(self.task_id, 540, 540)
        try:
            os.mkdir(self.directory)
        except Exception as e:
            print e
        self.particle_renderer = ParticleRenderer('shadow_map',
                                                  shadow_map_resolution=0.3, alpha=0.7, shadowing=2,
                                                  ambient_light=0.01,
                                                  light_direction=(1, 1, 0))
        self.resolution = kwargs['resolution']
        self.frame = 0

    def step(self, step_t):
        t = self.c.get_current_time()
        print 'Simulation time:', t
        T = time.time()
        self.c.step(step_t)
        print 'Time:', time.time() - T
        image_buffer = tc_core.Array2DVector3(self.video_manager.width, self.video_manager.height, Vector(0, 0, 0.0))
        particles = self.c.get_render_particles()
        particles.write(self.directory + '/particles%05d.bin' % self.frame)
        res = map(float, self.resolution)
        camera = Camera('pinhole', origin=(0, res[1] * 0.4, res[2] * 1.4),
                        look_at=(0, -res[1] * 0.5, 0), up=(0, 1, 0), fov=70,
                        width=10, height=10)
        self.particle_renderer.set_camera(camera)
        self.particle_renderer.render(image_buffer, particles)
        img = image_buffer_to_ndarray(image_buffer)
        img = LDRDisplay(exposure=2.0, adaptive_exposure=False).process(img)
        show_image('Vis', img)
        self.video_manager.write_frame(img)
        self.frame += 1

    def get_directory(self):
        return self.directory

    def make_video(self):
        self.video_manager.make_video()
