import taichi as ti
import numpy as np


@ti.data_oriented
class Cloth:
    def __init__(self, N):
        self.N = N
        self.NF = 2 * N**2  # number of faces
        self.NV = (N + 1)**2  # number of vertices
        self.NE = 2 * N * (N + 1) + 2 * N * N  # numbser os edges
        self.pos = ti.Vector.field(2, ti.f64, self.NV)
        self.vel = ti.Vector.field(2, ti.f64, self.NV)
        self.force = ti.Vector.field(2, ti.f64, self.NV)
        self.invMass = ti.field(ti.f64, self.NV)

        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.Jx = ti.Matrix.field(2, 2, ti.f64,
                                  self.NE)  # Jacobian with respect to position
        self.Jv = ti.Matrix.field(2, 2, ti.f64,
                                  self.NE)  # Jacobian with respect to velocity
        self.rest_len = ti.field(ti.f64, self.NE)
        self.ks = 1000.0  # spring stiffness
        self.kd = 0.5  # damping constant

        self.gravity = ti.Vector([0.0, -2.0])
        self.init_pos()
        self.init_edges()

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.pos[k] = ti.Vector([i, j]) / self.N * 0.5 + ti.Vector(
                [0.25, 0.25])
            self.vel[k] = ti.Vector([0, 0])
            self.invMass[k] = 1.0
        self.invMass[self.N] = 0.0
        self.invMass[self.NV - 1] = 0.0

    @ti.kernel
    def init_edges(self):
        for i, j in ti.ndrange(self.N + 1, self.N):
            self.spring[i * self.N + j] = ti.Vector(
                [i * (self.N + 1) + j, i * (self.N + 1) + j + 1])
            self.rest_len[i * self.N +
                          j] = (self.pos[i * (self.N + 1) + j] -
                                self.pos[i * (self.N + 1) + j + 1]).norm()
        start = self.N * (self.N + 1)
        for i, j in ti.ndrange(self.N, self.N + 1):
            self.spring[start + i + j * self.N] = ti.Vector(
                [i * (self.N + 1) + j, i * (self.N + 1) + j + self.N + 1])
            self.rest_len[start + i + j * self.N] = (
                self.pos[i * (self.N + 1) + j] -
                self.pos[i * (self.N + 1) + j + self.N + 1]).norm()
        start = 2 * self.N * (self.N + 1)
        for i, j in ti.ndrange(self.N, self.N):
            self.spring[start + i * self.N + j] = ti.Vector(
                [i * (self.N + 1) + j, (i + 1) * (self.N + 1) + j + 1])
            self.rest_len[start + i * self.N +
                          j] = (self.pos[i * (self.N + 1) + j] -
                                self.pos[(i + 1) *
                                         (self.N + 1) + j + 1]).norm()
        start = 2 * self.N * (self.N + 1) + self.N * self.N
        for i, j in ti.ndrange(self.N, self.N):
            self.spring[start + i * self.N + j] = ti.Vector(
                [i * (self.N + 1) + j + 1, (i + 1) * (self.N + 1) + j])
            self.rest_len[start + i * self.N +
                          j] = (self.pos[i * (self.N + 1) + j + 1] -
                                self.pos[(i + 1) * (self.N + 1) + j]).norm()

    def display(self, gui, radius=5, color=0xffffff):
        lines = self.spring.to_numpy()
        pos = self.pos.to_numpy()
        edgeBegin = np.zeros(shape=(lines.shape[0], 2))
        edgeEnd = np.zeros(shape=(lines.shape[0], 2))
        for i in range(lines.shape[0]):
            idx1, idx2 = lines[i][0], lines[i][1]
            edgeBegin[i] = pos[idx1]
            edgeEnd[i] = pos[idx2]
        gui.lines(edgeBegin, edgeEnd, radius=2, color=0x0000ff)
        gui.circles(self.pos.to_numpy(), radius, color)

    @ti.func
    def clear_force(self):
        for i in self.force:
            if self.invMass[i] != 0.0:
                self.force[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        for i in self.force:
            if self.invMass[i] != 0.0:
                self.force[i] += self.gravity / self.invMass[i]

        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dis = pos2 - pos1
            force = self.ks * (dis.norm() -
                               self.rest_len[i]) * dis.normalized()
            self.force[idx1] += force
            self.force[idx2] -= force

    @ti.kernel
    def updatePosVel(self, h: ti.f64):
        for i in self.pos:
            if self.invMass[i] != 0.0:
                self.vel[i] += h * self.force[i] * self.invMass[i]
                self.pos[i] += h * self.vel[i]

    def update(self, h):
        self.compute_force()
        self.updatePosVel(h)


if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    h = 0.001
    cloth = Cloth(N=5)

    pause = False
    gui = ti.GUI('Implicit Mass Spring System')
    while gui.running:
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                pause = not pause

        if not pause:
            cloth.update(h)

        cloth.display(gui)
        gui.show()
