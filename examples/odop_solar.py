import taichi as ti


@ti.data_oriented
class SolarSystem:
    def __init__(self, n, dt):
        self.n = n
        self.dt = dt
        self.x = ti.Vector(2, dt=ti.f32, shape=n)
        self.v = ti.Vector(2, dt=ti.f32, shape=n)
        self.center = ti.Vector(2, dt=ti.f32, shape=())

    @staticmethod
    @ti.func
    def random_around(center, radius):
        # random number in [center - radius, center + radius)
        return center + radius * (ti.random() - 0.5) * 2

    @ti.kernel
    def initialize(self):
        for i in range(self.n):
            offset = ti.Vector([0.0, self.random_around(0.3, 0.15)])
            self.x[i] = self.center[None] + offset
            self.v[i] = [-offset[1], offset[0]]
            self.v[i] *= 1.5 / offset.norm()

    @ti.func
    def gravity(self, pos):
        offset = -(pos - self.center[None])
        return offset / offset.norm()**3

    @ti.kernel
    def integrate(self):
        for i in range(self.n):
            self.v[i] += self.dt * self.gravity(self.x[i])
            self.x[i] += self.dt * self.v[i]


solar = SolarSystem(9, 0.0005)
solar.center[None] = [0.5, 0.5]
solar.initialize()

gui = ti.GUI("Solar System", background_color=0x25A6D9)

while True:
    if gui.get_event():
        if gui.event.key == gui.SPACE and gui.event.type == gui.PRESS:
            solar.initialize()
    for i in range(10):
        solar.integrate()
    gui.circle([0.5, 0.5], radius=20, color=0x8C274C)
    gui.circles(solar.x.to_numpy(), radius=5, color=0xFFFFFF)
    gui.show()
