import glob
import os

import pyglet
from pyglet.gl import *

from taichi.misc.util import get_os_name, get_unique_task_id
from taichi.misc.settings import get_output_directory

VIDEO_OUTPUT_ROOT = get_output_directory()


def normalized_color_255(*args):
    return tuple(map(lambda x: x / 255.0, args))


class SimulationWindow(pyglet.window.Window):
    def __init__(self, max_side, simulator, color_scheme, levelset_supersampling=2, show_grid=False, show_images=True,
                 rescale=True):
        if rescale:
            scale = min(1.0 * max_side / simulator.simulation_width, 1.0 * max_side / simulator.simulation_height)
            width = int(round(scale * simulator.simulation_width))
            height = int(round(scale * simulator.simulation_height))
        else:
            width = max_side
            height = max_side
        super(SimulationWindow, self).__init__(width=width, height=height, fullscreen=False, caption='Taichi',
                                               config=pyglet.gl.Config(sample_buffers=0, samples=0, depth_size=16,
                                                                       double_buffer=True))
        uuid = get_unique_task_id()
        self.video_filename = uuid + ".mp4"
        self.frame_output_path = VIDEO_OUTPUT_ROOT + '/' + uuid + '/'
        os.mkdir(self.frame_output_path[:-1])
        self.simulator = simulator
        self.frame_count = 0
        self.color_scheme = color_scheme
        self.show_images = show_images
        self.levelset_supersampling = levelset_supersampling
        self.show_grid = show_grid
        self.quit_pressed = False
        self.timer = 0
        self.next_video_checkpoint = 4
        pyglet.clock.schedule_interval(self.update, 1 / 120.0)
        pyglet.app.run()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            exit(0)
        if symbol == pyglet.window.key.ESCAPE:
            self.quit_pressed = True

    def clear_frames(self):
        files = glob.glob(self.frame_output_path + '*.png')
        for f in files:
            assert f.endswith('.png')
            os.remove(f)
        print '%d png frame files removed' % (len(files))
        os.rmdir(self.frame_output_path)

    def update(self, _):
        self.timer += 1
        print self.timer
        if self.timer % self.next_video_checkpoint == 0:
            self.make_video()
            self.next_video_checkpoint *= 2
        if not self.quit_pressed and not self.simulator.ended():
            self.simulator.step()
        else:
            exit(0)
        self.redraw()
        self.save_frame()

    def make_video(self):
        command = "ffmpeg -y -framerate 24 -i " + self.frame_output_path + \
                  "/frame%d.png -s:v " + str(self.width) + 'x' + str(self.height) + \
                  " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + VIDEO_OUTPUT_ROOT + \
                  "/" + self.video_filename
        print command

        os.system(command)
        # self.clear_frames()

    def save_frame(self):
        gl.glPixelTransferf(gl.GL_ALPHA_BIAS, 1.0)
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            self.frame_output_path + '/frame%d.png' % self.frame_count)
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
        label = pyglet.text.Label('t = %.5f' % (self.simulator.get_current_time()),
                                  font_name='Rockwell',
                                  font_size=12,
                                  x=10, y=self.height - 5,
                                  anchor_x='left', anchor_y='top')
        ls_width = self.width * self.levelset_supersampling
        ls_height = self.height * self.levelset_supersampling
        background_images, foreground_images = self.simulator.get_levelset_images(ls_width, ls_height, self.color_scheme)
        if self.show_images:
            for img in background_images:
                img.blit(0, 0, 0, self.width, self.height)
        if self.show_grid:
            self.render_grid()

        self.render_particles()

        if self.show_images:
            for img in foreground_images:
                img.blit(0, 0, 0, self.width, self.height)
        label.color = self.color_scheme['label']
        label.draw()

    def render_grid(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(0, 0, 0)
        glScalef(1.0 * self.width / self.simulator.config['simulation_width'],
                 1.0 * self.height / self.simulator.config['simulation_height'], 0)

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
                color = (int(p.color.x), int(p.color.y), int(p.color.z), 200)
            else:
                color = self.color_scheme['particles']
            for i in range(4):
                colors.append(color[i])
        pyglet.graphics.draw(len(particles), gl.GL_POINTS, ('v2f', positions), ('c4B', colors))
        glPopMatrix()
