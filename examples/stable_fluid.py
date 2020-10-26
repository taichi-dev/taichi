# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import taichi as ti
import numpy as np

use_mgpcg = False  # True to use multigrid-preconditioned conjugate gradients
res = 512
dt = 0.03
p_jacobi_iters = 160  # 40 for a quicker but less accurate result
f_strength = 10000.0
curl_strength = 0  # 7 for unrealistic visual enhancement
dye_decay = 0.99
force_radius = res / 3.0
debug = False
paused = False

ti.init(arch=ti.gpu)

if use_mgpcg:
    from mgpcg_advanced import MGPCG  # examples/mgpcg_advanced.py
    mgpcg = MGPCG(dim=2, N=res, n_mg_levels=6)

_velocities = ti.Vector.field(2, float, shape=(res, res))
_intermedia_velocities = ti.Vector.field(2, float, shape=(res, res))
_new_velocities = ti.Vector.field(2, float, shape=(res, res))
velocity_divs = ti.field(float, shape=(res, res))
velocity_curls = ti.field(float, shape=(res, res))
_pressures = ti.field(float, shape=(res, res))
_new_pressures = ti.field(float, shape=(res, res))
_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
_intermedia_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res))


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(res - 1, I))
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


@ti.func
def sample_minmax(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return min(a, b, c, d), max(a, b, c, d)


@ti.func
def backtrace_rk1(vf: ti.template(), p, dt: ti.template()):
    p -= dt * bilerp(vf, p)
    return p


@ti.func
def backtrace_rk2(vf: ti.template(), p, dt: ti.template()):
    p_mid = p - 0.5 * dt * bilerp(vf, p)
    p -= dt * bilerp(vf, p_mid)
    return p


@ti.func
def backtrace_rk3(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


backtrace = backtrace_rk3


@ti.kernel
def advect_semilag(vf: ti.template(), qf: ti.template(), new_qf: ti.template(),
                   intermedia_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p)


@ti.kernel
def advect_bfecc(vf: ti.template(), qf: ti.template(), new_qf: ti.template(),
                 intermedia_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        intermedia_qf[i, j] = bilerp(qf, p)

    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        # star means the temp value after a back tracing (forward advection)
        # two star means the temp value after a forward tracing (reverse advection)
        p_two_star = backtrace(vf, p, -dt)
        p_star = backtrace(vf, p, dt)
        q_star = intermedia_qf[i, j]
        new_qf[i, j] = bilerp(intermedia_qf, p_two_star)

        new_qf[i, j] = q_star + 0.5 * (qf[i, j] - new_qf[i, j])

        min_val, max_val = sample_minmax(qf, p_star)
        cond = min_val < new_qf[i, j] < max_val
        for k in ti.static(range(cond.n)):
            if not cond[k]:
                new_qf[i, j][k] = q_star[k]


advect = advect_bfecc


@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(),
                  imp_data: ti.ext_arr()):
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)
        momentum = mdir * f_strength * dt * factor
        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        dc = dyef[i, j]
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15)**2)) * ti.Vector(
                [imp_data[4], imp_data[5], imp_data[6]])
        dc *= dye_decay
        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j).x
        vr = sample(vf, i + 1, j).x
        vb = sample(vf, i, j - 1).y
        vt = sample(vf, i, j + 1).y
        vc = sample(vf, i, j)
        if i == 0:
            vl = 0
        if i == res - 1:
            vr = 0
        if j == 0:
            vb = 0
        if j == res - 1:
            vt = 0
        velocity_divs[i, j] = (vr - vl + vt - vb) * 0.5


@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j).y
        vr = sample(vf, i + 1, j).y
        vb = sample(vf, i, j - 1).x
        vt = sample(vf, i, j + 1).x
        vc = sample(vf, i, j)
        velocity_curls[i, j] = (vr - vl - vt + vb) * 0.5


@ti.kernel
def pressure_jacobi_single(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def pressure_jacobi_dual(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pcc = sample(pf, i, j)
        pll = sample(pf, i - 2, j)
        prr = sample(pf, i + 2, j)
        pbb = sample(pf, i, j - 2)
        ptt = sample(pf, i, j + 2)
        plb = sample(pf, i - 1, j - 1)
        prb = sample(pf, i + 1, j - 1)
        plt = sample(pf, i - 1, j + 1)
        prt = sample(pf, i + 1, j + 1)
        div = sample(velocity_divs, i, j)
        divl = sample(velocity_divs, i - 1, j)
        divr = sample(velocity_divs, i + 1, j)
        divb = sample(velocity_divs, i, j - 1)
        divt = sample(velocity_divs, i, j + 1)
        new_pf[i,
               j] = (pll + prr + pbb + ptt - divl - divr - divb - divt - div +
                     (plt + prt + prb + plb) * 2 + pcc * 4) * 0.0625


pressure_jacobi = pressure_jacobi_single

if pressure_jacobi == pressure_jacobi_dual:
    p_jacobi_iters //= 2


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    # anti-physics visual enhancement...
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb),
                           abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = min(max(vf[i, j] + force * dt, -1e3), 1e3)


def step(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt,
           _intermedia_velocities)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt,
           _intermedia_dye_buffer)
    velocities_pair.swap()
    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)

    divergence(velocities_pair.cur)

    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    if use_mgpcg:
        mgpcg.init(velocity_divs, -1)
        mgpcg.solve(max_iters=10)
        mgpcg.get_result(pressures_pair.cur)

    else:
        for _ in range(p_jacobi_iters):
            pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
            pressures_pair.swap()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)

    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f'divergence={div_s}')


class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)


gui = ti.GUI('Stable Fluid', (res, res))
md_gen = MouseDataGen()
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        if e.key == ti.GUI.ESCAPE:
            break
        elif e.key == 'r':
            paused = False
            reset()
        elif e.key == 'p':
            paused = not paused
        elif e.key == 'd':
            debug = not debug

    if not paused:
        mouse_data = md_gen(gui)
        step(mouse_data)

    gui.set_image(dyes_pair.cur)
    # To visualize velocity field:
    # gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
    # To visualize velocity divergence:
    # divergence(velocities_pair.cur); gui.set_image(velocity_divs.to_numpy() * 0.1 + 0.5)
    # To visualize velocity vorticity:
    # vorticity(velocities_pair.cur); gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
    gui.show()
