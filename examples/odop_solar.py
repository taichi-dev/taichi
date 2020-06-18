import taichi as ti
import math


@ti.data_oriented
class SolarSystem:
    def __init__(self, n, dt):
        # initializer of the solar system simulator
        self.n = n
        self.dt = dt
        self.x = ti.Vector(2, dt=ti.f32, shape=n)
        self.v = ti.Vector(2, dt=ti.f32, shape=n)
        self.center = ti.Vector(2, dt=ti.f32, shape=())

    @staticmethod
    @ti.func
    def random_vector_in(rmax):
        # create a random vector
        a = ti.random() * math.tau
        r = ti.random() * rmax
        return r * ti.Vector([ti.cos(a), ti.sin(a)])

    @ti.kernel
    def initialize(self):
        # initialization or reset
        for i in range(self.n):
            offset = self.random_vector_in(0.5)
            self.x[i] = self.center[None] + offset  # Offset from center
            self.v[i] = [-offset.y, offset.x]  # Perpendicular to offset
            self.v[i] += self.random_vector_in(0.02)  # Shaking
            self.v[i] *= 1 / offset.norm()**1.5  # Kepler's 3rd law

    @ti.func
    def gravity(self, pos):
        # compute gravitational acceleration at pos
        offset = -(pos - self.center[None])
        return offset / offset.norm()**3

    @ti.kernel
    def integrate(self):
        # semi-implicit time integration
        for i in range(self.n):
            self.v[i] += self.dt * self.gravity(self.x[i])
            self.x[i] += self.dt * self.v[i]

    def render(self, gui):
        # render the simulation scene on the GUI
        gui.circle([0.5, 0.5], radius=10, color=0xffaa88)
        gui.circles(solar.x.to_numpy(), radius=3, color=0xffffff)


solar = SolarSystem(8, 0.0001)
solar.center[None] = [0.5, 0.5]
solar.initialize()

gui = ti.GUI("Solar System", background_color=0x0071a)

while gui.running:
    # GUI event processing
    if gui.get_event(gui.PRESS):
        if gui.event.key == gui.SPACE:
            solar.initialize()
        elif gui.event.key == gui.ESCAPE:
            gui.running = False

    for i in range(10):
        solar.integrate()

    solar.render(gui)
    gui.show()
