import math

import taichi as ti

ti.init()


@ti.data_oriented
class SolarSystem:
    def __init__(self, n, dt):  # Initializer of the solar system simulator
        self.n = n
        self.dt = dt
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.center = ti.Vector.field(2, dtype=ti.f32, shape=())

    @staticmethod
    @ti.func
    def random_vector(radius):  # Create a random vector in circle
        theta = ti.random() * 2 * math.pi
        r = ti.random() * radius
        return r * ti.Vector([ti.cos(theta), ti.sin(theta)])

    @ti.kernel
    def initialize_particles(self):
        # (Re)initialize particle position/velocities
        for i in range(self.n):
            offset = self.random_vector(0.5)
            self.x[i] = self.center[None] + offset  # Offset from center
            self.v[i] = [-offset.y, offset.x]  # Perpendicular to offset
            self.v[i] += self.random_vector(0.02)  # Random velocity noise
            self.v[i] *= 1 / offset.norm() ** 1.5  # Kepler's third law

    @ti.func
    def gravity(self, pos):  # Compute gravity at pos
        offset = -(pos - self.center[None])
        return offset / offset.norm() ** 3

    @ti.kernel
    def integrate(self):  # Semi-implicit Euler time integration
        for i in range(self.n):
            self.v[i] += self.dt * self.gravity(self.x[i])
            self.x[i] += self.dt * self.v[i]

    @staticmethod
    def render(gui):  # Render the scene on GUI
        gui.circle([0.5, 0.5], radius=10, color=0xFFAA88)
        gui.circles(solar.x.to_numpy(), radius=3, color=0xFFFFFF)


def main():
    global solar

    solar = SolarSystem(8, 0.0001)
    solar.center[None] = [0.5, 0.5]
    solar.initialize_particles()

    gui = ti.GUI("Solar System", background_color=0x0071A)
    while gui.running:
        if gui.get_event() and gui.is_pressed(gui.SPACE):
            solar.initialize_particles()  # reinitialize when space bar pressed.

        for _ in range(10):  # Time integration
            solar.integrate()

        solar.render(gui)
        gui.show()


if __name__ == "__main__":
    main()
