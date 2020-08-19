import numpy as np
import taichi as ti
import time


@ti.data_oriented
class MGPCG:
    '''
Grid-based MGPCG solver for the possion equation:

    $\\nabla^2 x = k r$

Credits by `@KLozes <https://github.com/KLozes>`_.

See `examples/stable_fluid.py <https://github.com/taichi-dev/taichi/blob/master/examples/stable_fluid.py>`_ for usage example.

.. note::

    This solver only runs on CPU and CUDA backend since it requires the
    ``pointer`` SNode. Consider replace ``pointer`` by ``bitmasked`` on Metal,
    by ``dense`` on OpenGL.
    '''

    def __init__(self, dim=2, N=512, n_mg_levels=6):
        '''
        :parameter dim: Dimension of the fields.
        :parameter N: Resolution of the fields.
        :parameter n_mg_levels: Multigrid levels.
        '''

        # grid parameters
        self.use_multigrid = True

        self.N = N

        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 50
        self.dim = dim

        self.N_ext = self.N // 2  # number of ext cells set so that that total grid size is still power of 2
        self.N_tot = 2 * self.N

        # setup sparse simulation data arrays
        self.r = [ti.field(dtype=float)
                  for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.field(dtype=float)
                  for _ in range(self.n_mg_levels)]  # M^-1 self.r
        self.x = ti.field(dtype=float)  # solution
        self.p = ti.field(dtype=float)  # conjugate gradient
        self.Ap = ti.field(dtype=float)  # matrix-vector product
        self.alpha = ti.field(dtype=float)  # step size
        self.beta = ti.field(dtype=float)  # step size
        self.sum = ti.field(dtype=float)  # storage for reductions

        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [self.N_tot // 4]).dense(
            indices, 4).place(self.x, self.p, self.Ap)

        for l in range(self.n_mg_levels):
            self.grid = ti.root.pointer(indices,
                                        [self.N_tot // (4 * 2**l)]).dense(
                                            indices,
                                            4).place(self.r[l], self.z[l])

        ti.root.place(self.alpha, self.beta, self.sum)

    @ti.func
    def init_r(self, I, r_I):
        I = I + self.N_ext
        self.r[0][I] = r_I
        self.z[0][I] = 0
        self.Ap[I] = 0
        self.p[I] = 0
        self.x[I] = 0

    @ti.func
    def get_x(self, I):
        I = I + self.N_ext
        return self.x[I]

    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()):
        '''
        Specify quantity field before each solve step.

        :parameter r: (ti.field) Specify the quantity field
        :parameter k: (scalar) Specify the scale multiply to the quantity field

        $\\nabla^2 x = k r$
        '''
        for I in ti.grouped(ti.ndrange(*[self.N] * self.dim)):
            self.init_r(I, r[I] * k)

    @ti.kernel
    def get_result(self, x: ti.template()):
        '''
        Get the solution field after each solve step.

        :parameter x: (ti.field) Specify where to store the solution field

        $\\nabla^2 x = k r$
        '''
        for I in ti.grouped(ti.ndrange(*[self.N] * self.dim)):
            x[I] = self.get_x(I)

    @ti.func
    def neighbor_sum(self, x, I):
        ret = x[I] * 0
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            ret += x[I + offset] + x[I - offset]
        return ret

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            self.Ap[I] = 2 * self.dim * self.p[I] - self.neighbor_sum(
                self.p, I)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            res = self.r[l][I] - (2 * self.dim * self.z[l][I] -
                                  self.neighbor_sum(self.z[l], I))
            self.r[l + 1][I // 2] += res * 0.5

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] = self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase:
                self.z[l][I] = (self.r[l][I] + self.neighbor_sum(
                    self.z[l], I)) / (2 * self.dim)

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self, steps=-1, eps=1e-12, tol=1e-12, rel=1e-12):
        '''
        Solve the possion equation.
        Always use a ``for`` loop to iterate through me.

        :parameter steps: Specify the maximal steps, -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter tol: Specify the absolute tolerance of loss.
        :parameter rel: Specify the tolerance of loss relative to initial loss.

        :return: (generator) A sequence generator for the loss of each step.
        '''

        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        tol = max(tol, initial_rTr * rel)

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p
        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # CG
        while steps != 0:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]
            if rTr < tol:
                break

            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            yield rTr
            steps -= 1


class MGPCG_Example(MGPCG):
    def __init__(self):
        super().__init__(dim=3, N=128, n_mg_levels=4)

        self.N_gui = 512  # gui resolution

        self.pixels = ti.field(dtype=float,
                               shape=(self.N_gui, self.N_gui))  # image buffer

    @ti.kernel
    def init(self):
        for I in ti.grouped(ti.ndrange(*[self.N] * self.dim)):
            r_I = 5.0
            for k in ti.static(range(self.dim)):
                r_I *= ti.cos(5 * np.pi * I[k] / self.N)
            self.init_r(I, r_I)

    @ti.kernel
    def paint(self):
        if ti.static(self.dim == 3):
            kk = self.N_tot * 3 // 8
            for i, j in self.pixels:
                ii = int(i * self.N / self.N_gui) + self.N_ext
                jj = int(j * self.N / self.N_gui) + self.N_ext
                self.pixels[i, j] = self.x[ii, jj, kk] / self.N_tot

    def run(self):
        gui = ti.GUI("Multigrid Preconditioned Conjugate Gradients",
                     res=(self.N_gui, self.N_gui))

        self.init()
        for i, rTr in enumerate(self.solve(steps=400)):
            print(f'iter {i}, residual={rTr}')
            self.paint()
            gui.set_image(self.pixels)
            gui.show()

        ti.kernel_profiler_print()


if __name__ == '__main__':
    ti.init(kernel_profiler=True)
    solver = MGPCG_Example()
    t = time.time()
    solver.run()
    print(f'Solver time: {time.time() - t:.3f} s')
