from os import initgroups
import taichi as ti
import numpy as np
from taichi.lang import init

@ti.data_oriented
class Cloth:
    def __init__(self, N):
        self.N = N
        self.NF = 2 * N**2  # number of faces
        self.NV = (N + 1)**2  # number of vertices
        self.NE = 2 * N * (N + 1) + 2 * N * N  # numbser os edges
        self.pos = ti.Vector.field(2, ti.f32, self.NV)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV)
        self.vel = ti.Vector.field(2, ti.f32, self.NV)
        self.force = ti.Vector.field(2, ti.f32, self.NV)
        self.invMass = ti.field(ti.f32, self.NV)

        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.Jx = ti.Matrix.field(2, 2, ti.f32,
                                  self.NE)  # Jacobian with respect to position
        self.Jv = ti.Matrix.field(2, 2, ti.f32,
                                  self.NE)  # Jacobian with respect to velocity
        self.rest_len = ti.field(ti.f32, self.NE)
        self.ks = 1000.0  # spring stiffness
        self.kd = 0.5  # damping constant
        self.kf = 1.0e5 # fix point stiffness

        self.gravity = ti.Vector([0.0, -2.0])
        self.init_pos()
        self.init_edges()
        # Mass Matrix Builder, Damping Matrix Builder, Stiffness Matrix Builder
        self.MassBuilder = ti.SparseMatrixBuilder(2 * self.NV,
                                                  2 * self.NV,
                                                  max_num_triplets=10000)

        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()
        self.fix_vertex = [self.N, self.NV - 1]
        self.Jf = ti.Matrix.field(2,2, ti.f32, len(self.fix_vertex))
        
    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.pos[k] = ti.Vector([i, j]) / self.N * 0.5 + ti.Vector(
                [0.25, 0.25])
            self.initPos[k] = self.pos[k]
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

    @ti.kernel
    def init_mass_sp(self, M: ti.sparse_matrix_builder()):
        for i in range(self.NV):
            if self.invMass[i] != 0.0:
                mass = 1.0 / self.invMass[i]
                M[2 * i + 0, 2 * i + 0] += mass
                M[2 * i + 1, 2 * i + 1] += mass

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
        
        self.force[self.N] += self.kf * (self.initPos[self.N] - self.pos[self.N])
        self.force[self.NV - 1] += self.kf * (self.initPos[self.NV - 1] - self.pos[self.NV-1])

    @ti.kernel
    def compute_Jacobians(self):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1]],
                               [dx[1] * dx[0], dx[1] * dx[1]]])
            l = dx.norm()
            if l != 0.0:
                l = 1.0 / l
            self.Jx[i] = (I - self.rest_len[i] * l *
                          (I - dxtdx * l**2)) * self.ks
            self.Jv[i] = self.kd * I

        self.Jf[0] = ti.Matrix([[self.kf,0],[0, self.kf]])
        self.Jf[1] = ti.Matrix([[self.kf,0],[0, self.kf]])

    @ti.kernel
    def assemble_K(self, K: ti.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            K[2 * idx1 + 0, 2 * idx1 + 0] -= self.Jx[i][0, 0]
            K[2 * idx1 + 0, 2 * idx1 + 1] -= self.Jx[i][0, 1]
            K[2 * idx1 + 1, 2 * idx1 + 0] -= self.Jx[i][1, 0]
            K[2 * idx1 + 1, 2 * idx1 + 1] -= self.Jx[i][1, 1]

            K[2 * idx1 + 0, 2 * idx2 + 0] += self.Jx[i][0, 0]
            K[2 * idx1 + 0, 2 * idx2 + 1] += self.Jx[i][0, 1]
            K[2 * idx1 + 1, 2 * idx2 + 0] += self.Jx[i][1, 0]
            K[2 * idx1 + 1, 2 * idx2 + 1] += self.Jx[i][1, 1]

            K[2 * idx2 + 0, 2 * idx1 + 0] += self.Jx[i][0, 0]
            K[2 * idx2 + 0, 2 * idx1 + 1] += self.Jx[i][0, 1]
            K[2 * idx2 + 1, 2 * idx1 + 0] += self.Jx[i][1, 0]
            K[2 * idx2 + 1, 2 * idx1 + 1] += self.Jx[i][1, 1]

            K[2 * idx2 + 0, 2 * idx2 + 0] -= self.Jx[i][0, 0]
            K[2 * idx2 + 0, 2 * idx2 + 1] -= self.Jx[i][0, 1]
            K[2 * idx2 + 1, 2 * idx2 + 0] -= self.Jx[i][1, 0]
            K[2 * idx2 + 1, 2 * idx2 + 1] -= self.Jx[i][1, 1]
        # fix point constraint hessian
        K[2 * self.N + 0, 2 * self.N + 0] += self.Jf[0][0,0]
        K[2 * self.N + 0, 2 * self.N + 1] += self.Jf[0][0,1]
        K[2 * self.N + 1, 2 * self.N + 0] += self.Jf[0][1,0]
        K[2 * self.N + 1, 2 * self.N + 1] += self.Jf[0][1,1]

        K[2 * (self.NV - 1) + 0, 2 * (self.NV - 1) + 0] += self.Jf[1][0,0]
        K[2 * (self.NV - 1) + 0, 2 * (self.NV - 1) + 1] += self.Jf[1][0,1]
        K[2 * (self.NV - 1) + 1, 2 * (self.NV - 1) + 0] += self.Jf[1][1,0]
        K[2 * (self.NV - 1) + 1, 2 * (self.NV - 1) + 1] += self.Jf[1][1,1]

    @ti.kernel
    def assemble_D(self, D: ti.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            D[2 * idx1 + 0, 2 * idx1 + 0] -= self.Jv[i][0, 0]
            D[2 * idx1 + 0, 2 * idx1 + 1] -= self.Jv[i][0, 1]
            D[2 * idx1 + 1, 2 * idx1 + 0] -= self.Jv[i][1, 0]
            D[2 * idx1 + 1, 2 * idx1 + 1] -= self.Jv[i][1, 1]

            D[2 * idx1 + 0, 2 * idx2 + 0] += self.Jv[i][0, 0]
            D[2 * idx1 + 0, 2 * idx2 + 1] += self.Jv[i][0, 1]
            D[2 * idx1 + 1, 2 * idx2 + 0] += self.Jv[i][1, 0]
            D[2 * idx1 + 1, 2 * idx2 + 1] += self.Jv[i][1, 1]

            D[2 * idx2 + 0, 2 * idx1 + 0] += self.Jv[i][0, 0]
            D[2 * idx2 + 0, 2 * idx1 + 1] += self.Jv[i][0, 1]
            D[2 * idx2 + 1, 2 * idx1 + 0] += self.Jv[i][1, 0]
            D[2 * idx2 + 1, 2 * idx1 + 1] += self.Jv[i][1, 1]

            D[2 * idx2 + 0, 2 * idx2 + 0] -= self.Jv[i][0, 0]
            D[2 * idx2 + 0, 2 * idx2 + 1] -= self.Jv[i][0, 1]
            D[2 * idx2 + 1, 2 * idx2 + 0] -= self.Jv[i][1, 0]
            D[2 * idx2 + 1, 2 * idx2 + 1] -= self.Jv[i][1, 1]

    @ti.kernel
    def updatePosVel(self, h: ti.f32, dv: ti.ext_arr()):
        for i in self.pos:
            if self.invMass[i] != 0.0:
                self.vel[i] += ti.Vector([dv[2 * i], dv[2 * i + 1]])
                self.pos[i] += h * self.vel[i]


    def update(self, h):
        self.compute_force()

        self.compute_Jacobians()
        # Assemble global system
        DBuilder = ti.SparseMatrixBuilder(2 * self.NV,
                                          2 * self.NV,
                                          max_num_triplets=10000)
        self.assemble_D(DBuilder)
        D = DBuilder.build()

        KBuilder = ti.SparseMatrixBuilder(2 * self.NV,
                                          2 * self.NV,
                                          max_num_triplets=10000)
        self.assemble_K(KBuilder)
        K = KBuilder.build()

        A = self.M - h * D - h**2 * K

        vel = self.vel.to_numpy().reshape(2 * self.NV)
        force = self.force.to_numpy().reshape(2 * self.NV)
        b = (force + h * K @ vel) * h
        # Sparse solver
        solver = ti.SparseSolver(solver_type="LU")
        solver.analyze_pattern(A)
        solver.factorize(A)
        # Solve the linear system
        dv = solver.solve(b)
        self.updatePosVel(h, dv)


if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    h = 0.01
    cloth = Cloth(N=5)

    pause = True
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
