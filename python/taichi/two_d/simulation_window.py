import glob
import os
import time

import pyglet
from pyglet.gl import *

from taichi.misc.util import get_os_name, get_unique_task_id
from taichi.misc.settings import get_output_directory
from taichi.tools.video import VideoManager
import numpy as np


def normalized_color_255(*args):
    return tuple(map(lambda x: x / 255.0, args))


class SimulationWindow(pyglet.window.Window):
    def __init__(self, max_side, simulator, color_scheme, levelset_supersampling=2, show_grid=False, show_images=True,
                 rescale=True, video_framerate=24, video_output=True, substep=False, need_press=False, show_stat=True):
        if rescale:
            scale = min(1.0 * max_side / simulator.res[0], 1.0 * max_side / simulator.res[1])
            width = int(round(scale * simulator.res[0]))
            height = int(round(scale * simulator.res[1]))
        else:
            width = max_side
            height = max_side

        super(SimulationWindow, self).__init__(width=width, height=height, fullscreen=False, caption='Taichi',
                                               config=pyglet.gl.Config(sample_buffers=0, samples=0, depth_size=16,
                                                                       double_buffer=True))
        self.width = width
        self.height = height
        self.video_framerate = video_framerate
        self.task_id = get_unique_task_id()
        self.simulator = simulator
        self.frame_count = 0
        self.color_scheme = color_scheme
        self.show_images = show_images
        self.levelset_supersampling = levelset_supersampling
        self.show_grid = show_grid
        self.quit_pressed = False
        self.output_directory = os.path.join(get_output_directory(), self.task_id)
        self.cpu_time = 0
        self.show_stat = show_stat
        os.mkdir(self.output_directory)
        self.substep = substep
        self.video_output = video_output
        self.video_manager = VideoManager(self.output_directory, automatic_build=self.video_output)
        self.need_press = need_press
        self.pressed = False
        pyglet.clock.schedule_interval(self.update, 1 / 120.0)
        pyglet.app.run()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.J:
            self.pressed = True
        if symbol == pyglet.window.key.K:
            self.need_press = not self.need_press
        if symbol == pyglet.window.key.Q:
            exit(0)
        if symbol == pyglet.window.key.ESCAPE:
            self.quit_pressed = True

    def update(self, _):
        if not self.quit_pressed and not self.simulator.ended():
            if not self.need_press or self.pressed:
                t = time.time()
                if self.substep:
                    self.simulator.step(True)
                else:
                    self.simulator.step()
                ela_t = time.time() - t
                self.cpu_time += ela_t
                print 'CPU Time: %.3f [%.3f per frame]' % (ela_t, self.get_time_per_frame())
            self.pressed = False
        else:
            if self.video_output:
                self.video_manager.make_video()
            exit(0)
        self.redraw()
        self.save_frame()

    def save_frame(self):
        gl.glPixelTransferf(gl.GL_ALPHA_BIAS, 1.0)
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        data = buffer.get_image_data().data
        img = (np.fromstring(data, dtype=np.uint8).reshape((self.height, self.width, 4)) / 255.0).astype(
            np.float32).swapaxes(0, 1)
        self.video_manager.write_frame(img[:, :, :])
        gl.glPixelTransferf(gl.GL_ALPHA_BIAS, 0.0)
        self.frame_count += 1

    def redraw(self):
        glClearColor(*normalized_color_255(*self.color_scheme['background']))
        self.clear()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        background = self.simulator.get_background_image(self.width, self.height)
        if background:
            background.blit(0, 0, 0, self.width, self.height)
        ls_width = self.width * self.levelset_supersampling
        ls_height = self.height * self.levelset_supersampling
        background_images, foreground_images = self.simulator.get_levelset_images(ls_width, ls_height,
                                                                                  self.color_scheme)
        if self.show_images:
            for img in background_images:
                img.blit(0, 0, 0, self.width, self.height)
        if self.show_grid:
            self.render_grid()

        self.render_particles()

        if self.show_images:
            for img in foreground_images:
                img.blit(0, 0, 0, self.width, self.height)

        if self.show_stat:
            label = pyglet.text.Label('t = %.5f' % (self.simulator.get_current_time()),
                                      font_name='Rockwell',
                                      font_size=12,
                                      x=10, y=20,
                                      anchor_x='left', anchor_y='top')
            label.color = self.color_scheme['label']
            label.draw()
            label = pyglet.text.Label('total time = %.2fs' % self.cpu_time,
                                      font_name='Rockwell',
                                      font_size=12,
                                      x=10, y=40,
                                      anchor_x='left', anchor_y='top')
            label.color = self.color_scheme['label']
            label.draw()
            label = pyglet.text.Label('per frame = %.3fs' % self.get_time_per_frame(),
                                      font_name='Rockwell',
                                      font_size=12,
                                      x=10, y=60,
                                      anchor_x='left', anchor_y='top')
            label.color = self.color_scheme['label']
            label.draw()

    def get_time_per_frame(self):
        return self.cpu_time / (1 + self.frame_count)

    def render_grid(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(0, 0, 0)
        glScalef(1.0 * self.width / self.simulator.config['res'][0],
                 1.0 * self.height / self.simulator.config['res'][1], 0)

        line_num_x = self.simulator.resolution[0]
        line_num_y = self.simulator.resolution[1]
        positions = []
        for i in range(1, line_num_x):
            positions.append(1.0 * i)
            positions.append(0.0)
            positions.append(1.0 * i)
            positions.append(line_num_y)
        for i in range(1, line_num_y):
            positions.append(0.0)
            positions.append(1.0 * i)
            positions.append(line_num_x)
            positions.append(1.0 * i)

        points = 2 * (line_num_x + line_num_y - 2)
        pyglet.graphics.draw(points, gl.GL_LINES, ('v2f', positions), ('c4B', [128, 128, 128, 50] * points))
        glPopMatrix()

    def render_particles(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(0, 0, 0)
        glScalef(1.0 * self.width / self.simulator.config['res'][0],
                 1.0 * self.height / self.simulator.config['res'][1], 0)

        particles = self.simulator.get_particles()

        positions = []
        colors = []
        glPointSize(1.0)
        for p in particles:
            positions.append(p.position.x)
            positions.append(p.position.y)
            if p.color.x != -1:
                color = p.color * 255
                color = tuple(map(int, (color.x, color.y, color.z, 200)))
            else:
                color = self.color_scheme['particles']
            for i in range(4):
                colors.append(color[i])
        print "#part.", len(positions)
        pyglet.graphics.draw(len(particles), gl.GL_POINTS, ('v2f', positions), ('c4B', colors))
        glPopMatrix()
