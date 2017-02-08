import pyglet
from pyglet.gl import *
from taichi.two_d.levelset_2d import LevelSet2D
from taichi import Vector


class TaichiWindow(pyglet.window.Window):
    def __init__(self, width, height):
        super(TaichiWindow, self).__init__(width=width, height=height, fullscreen=False, caption='Taichi',
                                          config=pyglet.gl.Config(sample_buffers=0, samples=0, double_buffer=True))
        self.levelset_resolution = 100
        self.scale = 1.0 / self.levelset_resolution
        self.levelset = LevelSet2D(self.levelset_resolution, self.levelset_resolution, self.scale)
        self.levelset.add_polygon([(0.25, 0.4), (0.7, 0.3), (0.3, 0.7)], False)
        # self.levelset.add_sphere(Vector(0.15, 0.2), 0.10, False)
        # self.levelset.add_sphere(Vector(200, 100), 90, False)
        # self.levelset.add_sphere(Vector(200, 200), 10, False)

    def update(self, delta_t):
        if self.simulator.get_current_time() < self.simulation_time:
            t = self.simulator.get_current_time()
            if self.events and t > self.events[0][0]:
                self.events[0][1](self.simulator)
                self.events = self.events[1:]
            self.simulator.step(self.time_step)
            self.particles = self.simulator.get_particles()
        else:
            self.make_video()
            exit(0)
        self.on_draw()
        self.save_frame()

    def on_draw(self):
        self.clear()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        positions = []
        colors = []
        for i in range(self.width):
            for j in range(self.height):
                positions.append(i)
                positions.append(j)
                pos = Vector((i + 0.5) / self.width, (j + 0.5) / self.height)
                c = self.levelset.get(pos) / 4.0 + 0.5
                c = min(1, max(c, 0))
                grad = self.levelset.get_normalized_gradient(pos)
                # grad = Vector(0, 0)
                colors.append(c)
                colors.append(grad.x / 2 + 0.5)
                colors.append(grad.y / 2 + 0.5)
        pyglet.graphics.draw(self.width * self.height, gl.GL_POINTS, ('v2i', positions), ('c3f', colors))


if __name__ == '__main__':
    window = TaichiWindow(400, 400)
    pyglet.app.run()
