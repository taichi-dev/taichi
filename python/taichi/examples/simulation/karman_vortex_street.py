# Fluid solver based on lattice boltzmann method using taichi language
# Author : Wang (hietwll@gmail.com)
# Original code at https://github.com/hietwll/LBM_Taichi

import matplotlib
import matplotlib.cm as cm
import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)


@ti.data_oriented
class lbm_solver:
    def __init__(
            self,
            nx,  # domain size
            ny,
            niu,  # viscosity of fluid
            bc_type,  # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
            bc_value,  # if bc_type = 0, we need to specify the velocity in bc_value
            cy=0,  # whether to place a cylindrical obstacle
            cy_para=[0.0, 0.0, 0.0],  # location and radius of the cylinder
    ):
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.mask = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.field(dtype=ti.i32, shape=(9, 2))
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.field(dtype=ti.f32, shape=(4, 2))
        self.cy = cy
        self.cy_para = ti.field(dtype=ti.f32, shape=3)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))
        self.cy_para.from_numpy(np.array(cy_para, dtype=np.float32))
        arr = np.array([
            4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
        ],
                       dtype=np.float32)
        self.w.from_numpy(arr)
        arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]],
                       dtype=np.int32)
        self.e.from_numpy(arr)

    @ti.func  # compute equilibrium distribution function
    def f_eq(self, i, j, k):
        eu = ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0] + ti.cast(
            self.e[k, 1], ti.f32) * self.vel[i, j][1]
        uv = self.vel[i, j][0]**2.0 + self.vel[i, j][1]**2.0
        return self.w[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu**2 -
                                             1.5 * uv)

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            self.rho[i, j] = 1.0
            self.mask[i, j] = 0.0
            for k in ti.static(range(9)):
                self.f_new[i, j][k] = self.f_eq(i, j, k)
                self.f_old[i, j][k] = self.f_new[i, j][k]
            if (self.cy == 1):
                if ((ti.cast(i, ti.f32) - self.cy_para[0])**2.0 +
                    (ti.cast(j, ti.f32) - self.cy_para[1])**2.0 <=
                        self.cy_para[2]**2.0):
                    self.mask[i, j] = 1.0

    @ti.kernel
    def collide_and_stream(self):  # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.f_new[i,j][k] = (1.0-self.inv_tau)*self.f_old[ip,jp][k] + \
                                        self.f_eq(ip,jp,k)*self.inv_tau

    @ti.kernel
    def update_macro_var(self):  # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0.0
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j][0] += (ti.cast(self.e[k, 0], ti.f32) *
                                      self.f_new[i, j][k])
                self.vel[i, j][1] += (ti.cast(self.e[k, 1], ti.f32) *
                                      self.f_new[i, j][k])
            self.vel[i, j][0] /= self.rho[i, j]
            self.vel[i, j][1] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self):  # impose boundary conditions
        # left and right
        for j in ti.ndrange(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in ti.ndrange(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # cylindrical obstacle
        # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
        for i, j in ti.ndrange(self.nx, self.ny):
            if (self.cy == 1 and self.mask[i, j] == 1):
                self.vel[i, j][0] = 0.0  # velocity is zero at solid boundary
                self.vel[i, j][1] = 0.0
                inb = 0
                jnb = 0
                if (ti.cast(i, ti.f32) >= self.cy_para[0]):
                    inb = i + 1
                else:
                    inb = i - 1
                if (ti.cast(j, ti.f32) >= self.cy_para[1]):
                    jnb = j + 1
                else:
                    jnb = j - 1
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if (outer == 1):  # handle outer boundary
            if (self.bc_type[dr] == 0):
                self.vel[ibc, jbc][0] = self.bc_value[dr, 0]
                self.vel[ibc, jbc][1] = self.bc_value[dr, 1]
            elif (self.bc_type[dr] == 1):
                self.vel[ibc, jbc][0] = self.vel[inb, jnb][0]
                self.vel[ibc, jbc][1] = self.vel[inb, jnb][1]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        for k in ti.static(range(9)):
            self.f_old[ibc,jbc][k] = self.f_eq(ibc,jbc,k) - self.f_eq(inb,jnb,k) + \
                                        self.f_old[inb,jnb][k]

    def solve(self):
        gui = ti.GUI('lbm solver', (self.nx, 2 * self.ny))
        self.init()
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            for _ in range(10):
                self.collide_and_stream()
                self.update_macro_var()
                self.apply_bc()

            ##  code fragment displaying vorticity is contributed by woclass
            vel = self.vel.to_numpy()
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            vel_mag = (vel[:, :, 0]**2.0 + vel[:, :, 1]**2.0)**0.5
            ## color map
            colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0),
                      (0.176, 0.976, 0.529), (0, 1, 1)]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'my_cmap', colors)
            vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                vmin=-0.02, vmax=0.02),
                                        cmap=my_cmap).to_rgba(vor)
            vel_img = cm.plasma(vel_mag / 0.15)
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()

    def pass_to_py(self):
        return self.vel.to_numpy()[:, :, 0]


if __name__ == '__main__':
    lbm = lbm_solver(801, 201, 0.01, [0, 0, 1, 0],
                     [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 1,
                     [160.0, 100.0, 20.0])
    lbm.solve()
