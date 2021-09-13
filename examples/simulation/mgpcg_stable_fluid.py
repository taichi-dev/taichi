# A mgpcg + reflection enhanced version for https://github.com/taichi-dev/taichi/blob/master/examples/simulation/stable_fluid.py
# By @Mingrui Zhang @Dunfan Lu

import numpy as np

import taichi as ti


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


# Visualization control
@ti.data_oriented
class Dye:
    def __init__(self, res, advect_kernel) -> None:
        self.res = res

        self.dye_decay = 0.99
        self.f_strength = 10000.0
        self.force_radius = np.mean(res) / 5.0
        self.dye_buffer = ti.Vector.field(3, float, shape=res)
        self.new_dye_buffer = ti.Vector.field(3, float, shape=res)
        self.has_dye = ti.field(int, shape=())
        self.dyes_pair = TexPair(self.dye_buffer, self.new_dye_buffer)
        self.interactive_render = ti.Vector.field(3, float, shape=res)
        self.advect_kernel = advect_kernel

        self.source_colors = dict()
        self.initialized = False

    def init(self):
        self.initialized = True

    @ti.kernel
    def apply_impulse(self, vf: ti.template(), dyef: ti.template(),
                      imp_data: ti.ext_arr(), dt: ti.template(),
                      mean_res: float):

        for i, j in vf:
            dir = ti.Vector([imp_data[0], imp_data[1]])
            dir_norm = dir.norm() + 1e-5

            endpoint = ti.Vector([imp_data[2], imp_data[3]])
            cell_point = ti.Vector([i + 0.5, j + 0.5])
            distance_sqr = (cell_point - endpoint).norm_sqr()

            factor = ti.exp(-distance_sqr / self.force_radius)
            momentum = (dir / dir_norm * self.f_strength * factor) * dt
            v = vf[i, j]
            vf[i, j] = v + momentum

            # add dye
            max_dye = 1
            dye_color = dyef[i, j]
            if dir.norm() > 0.5:
                dye_color += ti.exp(
                    -distance_sqr * (4 / (mean_res / 15)**2)) * dir_norm / (
                        mean_res / 15.0) * ti.Vector(
                            [imp_data[4], imp_data[5], imp_data[6]])
                dye_color[0] = min(dye_color[0], max_dye)
                dye_color[1] = min(dye_color[1], max_dye)
                dye_color[2] = min(dye_color[2], max_dye)
            dyef[i, j] = dye_color

    @ti.kernel
    def apply_dye_decay(self, dyef: ti.template()):
        for i, j in dyef:
            dyef[i, j] = dyef[i, j] * self.dye_decay

    @ti.kernel
    def check_dye(self, dye: ti.template()):
        self.has_dye[None] = 0
        for i, j in dye:
            threshold = 0.01
            if dye[i, j][0] > threshold or dye[i, j][1] > threshold or dye[
                    i, j][2] > threshold:
                self.has_dye[None] = 1

    def apply_control(self, control_data, velocity_feild, dt):
        for data in control_data:
            source_id = data[4]
            if source_id not in self.source_colors:
                self.source_colors[source_id] = (np.random.rand(3) * 0.7) + 0.3
                self.source_colors[source_id] = np.array(
                    self.source_colors[source_id], dtype=np.float32)
            impulse_data = data[:4]
            color = self.source_colors[source_id]
            self.apply_impulse(velocity_feild, self.dyes_pair.cur,
                               np.concatenate([impulse_data, color]), dt,
                               np.mean(self.res))
        ids_to_remove = []
        for source_id in self.source_colors:
            found = False
            for data in control_data:
                if data[4] == source_id:
                    found = True
            if not found:
                ids_to_remove.append(source_id)
        for id in ids_to_remove:
            del self.source_colors[id]

    def step(self, velocity_feild, dt):
        self.advect_kernel(velocity_feild, self.dyes_pair.cur,
                           self.dyes_pair.nxt, dt)
        self.dyes_pair.swap()
        self.apply_dye_decay(self.dyes_pair.cur)

    def finalize_step(self, velocity_feild):
        self.check_dye(self.dyes_pair.cur)
        if self.has_dye[None] == 0:
            velocity_feild.fill(0)
            self.dyes_pair.cur.fill(0)

    def visualize_dye(self):
        return self.dyes_pair.cur


@ti.data_oriented
class MGPCG:
    def __init__(self,
                 dim=2,
                 resolution=(512, 512),
                 offset=None,
                 n_mg_levels=5,
                 block_size=8,
                 real=ti.f32,
                 use_multigrid=True,
                 sparse=False):
        self.use_multigrid = use_multigrid
        assert len(resolution) == dim
        self.res = resolution
        self.N_multigrid = []
        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10
        self.dim = dim
        self.real = real
        self.sparse = sparse

        if offset is None:
            self.offset = [-n // 2 for n in self.res]
        else:
            self.offset = offset
            assert len(offset) == self.dim

        self.r = [ti.field(dtype=self.real) for _ in range(self.n_mg_levels)]
        self.z = [ti.field(dtype=self.real) for _ in range(self.n_mg_levels)]

        self.x = ti.field(dtype=self.real)
        self.p = ti.field(dtype=self.real)
        self.Ap = ti.field(dtype=self.real)
        self.alpha = ti.field(dtype=self.real, shape=())
        self.beta = ti.field(dtype=self.real, shape=())
        self.sum = ti.field(dtype=self.real, shape=())
        self.old_zTr = ti.field(dtype=self.real, shape=())
        self.new_zTr = ti.field(dtype=self.real, shape=())

        indices = ti.ijk if self.dim == 3 else ti.ij

        coarsened_offset = self.offset
        coarsened_grid_size = list(self.res)
        self.grids = []

        for l in range(self.n_mg_levels):
            self.N_multigrid.append(coarsened_grid_size)
            sparse_grid_size = [
                dim_size + block_size * 2 for dim_size in coarsened_grid_size
            ]
            sparse_grid_offset = [o - block_size for o in coarsened_offset]
            print(f'Level {l}')
            print(f'  coarsened_grid_size {coarsened_grid_size}')
            print(f'  coarsened_offset {coarsened_offset}')

            grid = None
            if sparse:
                grid = ti.root.pointer(
                    indices,
                    [dim_size // block_size for dim_size in sparse_grid_size])
            else:
                grid = ti.root.dense(indices, [
                    dim_size // block_size for dim_size in coarsened_grid_size
                ])

            fields = []

            if l == 0:
                # Finest grid
                fields += [self.x, self.p, self.Ap]
            fields += [self.r[l], self.z[l]]

            if sparse:
                for f in fields:
                    grid.dense(indices, block_size).place(f)
            else:
                for f in fields:
                    grid.dense(indices, block_size).place(f)

            self.grids.append(grid)

            new_coarsened_offset = []
            for o in coarsened_offset:
                new_coarsened_offset.append(o // 2)
            coarsened_offset = new_coarsened_offset

            new_coarsened_grid_size = []
            for d in coarsened_grid_size:
                new_coarsened_grid_size.append(d // 2)
            coarsened_grid_size = new_coarsened_grid_size

    @ti.func
    def init_r(self, I, r_I):
        self.r[0][I] = r_I
        self.z[0][I] = 0
        self.Ap[I] = 0
        self.p[I] = 0
        self.x[I] = 0

    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()):
        for I in ti.grouped(r):
            self.init_r(I, r[I] * k)

    @ti.kernel
    def fetch_result(self, x: ti.template()):
        for I in ti.grouped(x):
            x[I] = self.x[I]

    @ti.func
    def sample(self, x, I):
        res = ti.Vector(x.shape)
        # Add Neumann boundary condition
        II = ti.max(0, ti.min(res - 1, I))
        for D in ti.static(range(self.dim)):
            II[D] = ti.assume_in_range(II[D], I[D], -1, 1)
        return x[II]

    @ti.func
    def neighbor_sum(self, x, I):
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            ret += self.sample(x, I + offset) + self.sample(x, I - offset)
        return ret

    @ti.kernel
    def compute_Ap(self):
        # Enable block local for sparse allocation
        if ti.static(self.sparse):
            ti.block_local(self.p)
        for I in ti.grouped(self.Ap):
            self.Ap[I] = 2 * self.dim * self.p[I] - self.neighbor_sum(
                self.p, I)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        ti.block_dim(32)
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def pre_multiply(self, p: ti.template(), q: ti.template()):
        for I in ti.grouped(p):
            self.pre_multiply_cache[I] = p[I] * q[I]

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
    def compute_alpha(self, eps: ti.template()):
        self.sum[None] = 0.0
        ti.block_dim(32)
        for I in ti.grouped(self.p):
            self.sum[None] += self.p[I] * self.Ap[I]
        self.alpha[None] = self.old_zTr[None] / max(self.sum[None], eps)

    @ti.kernel
    def compute_rTr(self, iter: ti.i32, verbose: ti.template()) -> ti.f32:
        rTr = 0.0
        ti.block_dim(32)
        for I in ti.grouped(self.r[0]):
            rTr += self.r[0][I] * self.r[0][I]
        if verbose:
            print('iter', iter, '|residual|_2=', ti.sqrt(rTr))
        return rTr

    @ti.kernel
    def compute_beta(self, eps: ti.template()):
        # beta = new_rTr / old_rTr
        self.new_zTr[None] = 0
        ti.block_dim(32)
        for I in ti.grouped(self.r[0]):
            self.new_zTr[None] += self.z[0][I] * self.r[0][I]
        self.beta[None] = self.new_zTr[None] / max(self.old_zTr[None], eps)

    @ti.kernel
    def update_zTr(self):
        self.old_zTr[None] = self.new_zTr[None]

    @ti.kernel
    def restrict(self, l: ti.template()):
        # Enable block local for sparse allocation
        if ti.static(self.sparse):
            ti.block_local(self.z[l])
        for I in ti.grouped(self.r[l]):
            residual = self.r[l][I] - (2 * self.dim * self.z[l][I] -
                                       self.neighbor_sum(self.z[l], I))
            self.r[l + 1][I // 2] += residual * 0.5

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] = self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red-black Gauss-Seidel phase
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

    def solve(self,
              max_iters=-1,
              eps=1e-12,
              abs_tol=1e-12,
              rel_tol=1e-12,
              iter_batch_size=2,
              verbose=False):

        self.reduce(self.r[0], self.r[0])
        residuals = []
        initial_rTr = self.sum[None]
        residuals.append(initial_rTr)

        tol = max(abs_tol, initial_rTr * rel_tol)

        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()
        self.compute_beta(eps)
        self.update_zTr()

        # Conjugate gradients
        iter = 0
        while max_iters == -1 or iter < max_iters:
            self.compute_Ap()
            self.compute_alpha(eps)

            # x += alpha p
            self.update_x()

            # r -= -alpha A p
            self.update_r()

            # ti.async_flush()
            if iter % iter_batch_size == iter_batch_size - 1:
                rTr = self.compute_rTr(iter, verbose)
                residuals.append(rTr)
                if rTr < tol:
                    break

            # z = M^-1 r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            self.compute_beta(eps)
            # p = z + beta p
            self.update_p()
            self.update_zTr()

            iter += 1
        return residuals


class MouseDataGen(object):
    def __init__(self, res):
        self.prev_mouse = None
        self.prev_id = 0
        self.res = res

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:5]: id
        mouse_data = np.zeros(5, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            # mxy: [x, y] in matrix coordinates i.e.:
            # y
            # ^
            # |
            # |-------> x
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * np.array(
                self.res)
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                self.prev_id += 1
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4] = self.prev_id
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None

        return np.array([mouse_data])


def resolution_checker(res, block_size):
    # Padded the resolution so that can be divided by the block_size
    padded_res = []
    for r in res:
        assert r > block_size
        remainder = r % block_size
        if remainder != 0:
            r += block_size - remainder
        padded_res.append(r)
    print(f"Padded resolution: {padded_res}")
    return tuple(padded_res)


dim = 2
block_size = 16

res = (512, 512)

res = resolution_checker(res, block_size)

dt = 0.03
p_jacobi_iters = 500
p_mgpcg_iters = 5

debug = False
paused = False
use_mgpcg = True
use_mg = True
use_reflection = False
use_interactive = True

frame_id = 0

ti.init(arch=ti.gpu, kernel_profiler=True)

_velocities = ti.Vector.field(2, float, shape=res)
_new_velocities = ti.Vector.field(2, float, shape=res)
velocity_divs = ti.field(float, shape=res)
_pressures = ti.field(float, shape=res)
_new_pressures = ti.field(float, shape=res)

res_vec = ti.Vector(res)

mg_sovler = None

mg_sovler = MGPCG(dim=2,
                  resolution=res,
                  block_size=block_size,
                  n_mg_levels=3,
                  use_multigrid=use_mg,
                  sparse=False)

velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)


@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(res_vec - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


# 3rd order Runge-Kutta
@ti.func
def backtrace(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def init_pressure_solver(v: ti.template()):
    for I in ti.grouped(v):
        mg_sovler.init_r(I, v[I])


@ti.func
def semi_lagrance(vf: ti.template(), qf: ti.template(), x: ti.template(),
                  dt: ti.template()):
    old_x = backtrace(vf, x, dt)
    q = bilerp(qf, old_x)
    return q


@ti.func
def maccormack(vf: ti.template(), qf: ti.template(), x: ti.template(),
               dt: ti.template()):
    q = bilerp(qf, x)

    old_x = backtrace(vf, x, dt)
    old_q = bilerp(qf, old_x)

    old_new_x = backtrace(vf, old_x, -dt)
    old_new_q = bilerp(qf, old_new_x)

    err = q - old_new_q

    return old_q + err * 0.5


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template(),
           dt: ti.template()):
    for I in ti.grouped(qf):
        new_qf[I] = maccormack(vf, qf, I + 0.5, dt)


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res[0] - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res[1] - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


arts = Dye(res, advect)


def run_advection(dt):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt, dt)
    velocities_pair.swap()


def compute_pressure():
    divergence(velocities_pair.cur)
    if use_mgpcg:
        mg_sovler.init(velocity_divs, -1)
        mg_sovler.solve(max_iters=p_mgpcg_iters, verbose=False)
        mg_sovler.fetch_result(pressures_pair.cur)
    else:
        for it in range(p_jacobi_iters):
            pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
            pressures_pair.swap()


def step(control_data):
    global frame_id

    if not arts.initialized:
        arts.init()

    if not use_reflection:
        arts.step(velocities_pair.cur, dt)
        run_advection(dt)
        arts.apply_control(control_data, velocities_pair.cur, dt)
        compute_pressure()
        subtract_gradient(velocities_pair.cur, pressures_pair.cur)
    elif use_reflection:
        arts.step(velocities_pair.cur, dt)
        run_advection(dt * 0.5)
        arts.apply_control(control_data, velocities_pair.cur, dt)
        compute_pressure()
        subtract_gradient(velocities_pair.cur, pressures_pair.cur)

        subtract_gradient(velocities_pair.cur, pressures_pair.cur)
        run_advection(dt * 0.5)
        compute_pressure()
        subtract_gradient(velocities_pair.cur, pressures_pair.cur)

    arts.finalize_step(velocities_pair.cur)
    frame_id += 1


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    arts.dyes_pair.cur.fill(0)


gui_window = None
canvas = None
press_token = None
escape_token = None
md_gen = MouseDataGen(res)

# GGUI requiares vulkan installed
use_ggui = False

if use_ggui:
    gui_window = ti.ui.Window('Stable Fluid', res, vsync=True)
    canvas = gui_window.get_canvas()
    press_token = ti.ui.PRESS
    escape_token = ti.ui.ESCAPE
else:
    gui_window = ti.GUI('Stable Fluid', res)
    press_token = ti.GUI.PRESS
    escape_token = ti.GUI.ESCAPE

control_data_gen = MouseDataGen(res=res)
visualize_d = True  #visualize dye (default)
visualize_v = False  #visualize velocity

while gui_window.running:
    if gui_window.get_event(press_token):
        e = gui_window.event
        if e.key == escape_token:
            break
        elif e.key == 'r':
            paused = False
            reset()
        elif e.key == 'v':
            visualize_v = True
            visualize_c = False
            visualize_d = False
        elif e.key == 'd':
            visualize_d = True
            visualize_v = False
            visualize_c = False
        elif e.key == 'p':
            paused = not paused
        elif e.key == 'd':
            debug = not debug
        elif e.key == 'h':
            # Reflection for higher quality vorticity
            if not use_reflection:
                use_reflection = True
            else:
                use_reflection = False

    if not paused:
        mouse_data = md_gen(gui_window)
        step(mouse_data)
    if visualize_d:
        dye_field = arts.visualize_dye()
        if use_ggui:
            canvas.set_image(dye_field)
        else:
            gui_window.set_image(dye_field)
    elif visualize_v:
        if use_ggui:
            canvas.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
        else:
            gui_window.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
    gui_window.show()
