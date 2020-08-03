# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import taichi as ti
import numpy as np
import time

res = 512 #600
dt = 0.03
p_jacobi_iters = 100 #30
f_strength = 10000.0
curl_strength = 0 #3
dye_decay = 0.99
force_radius = res / 3.0
debug = False
paused = False

ti.init(arch=ti.gpu)

_velocities = ti.Vector.field(2, ti.f32, shape=(res, res))
_new_velocities = ti.Vector.field(2, ti.f32, shape=(res, res))
velocity_divs = ti.field(ti.f32, shape=(res, res))
velocity_curls = ti.field(ti.f32, shape=(res, res))
_pressures = ti.field(ti.f32, shape=(res, res))
_new_pressures = ti.field(ti.f32, shape=(res, res))
_dye_buffer = ti.Vector.field(3, ti.f32, shape=(res, res))
_new_dye_buffer = ti.Vector.field(3, ti.f32, shape=(res, res))


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
    iu, iv = int(s), int(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu + 0.5, iv + 0.5)
    b = sample(vf, iu + 1.5, iv + 0.5)
    c = sample(vf, iu + 0.5, iv + 1.5)
    d = sample(vf, iu + 1.5, iv + 1.5)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.func
def sample_minmax(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = int(s), int(t)
    a = sample(vf, iu + 0.5, iv + 0.5)
    b = sample(vf, iu + 1.5, iv + 0.5)
    c = sample(vf, iu + 0.5, iv + 1.5)
    d = sample(vf, iu + 1.5, iv + 1.5)
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
    p -= dt * ((2/9) * v1 + (1/3) * v2 + (4/9) * v3)
    return p


backtrace = backtrace_rk3


@ti.kernel
def advect_semilag(vf: ti.template(),
        qf: ti.template(), new_qf: ti.template()):
    ti.cache_read_only(qf, vf)
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p)


@ti.kernel
def advect_bfecc(vf: ti.template(),
        qf: ti.template(), new_qf: ti.template()):
    ti.cache_read_only(qf, vf)
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p_mid = backtrace(vf, p, dt)
        q_mid = bilerp(qf, p_mid)
        p_fin = backtrace(vf, p_mid, -dt)
        q_fin = bilerp(qf, p_fin)
        new_qf[i, j] = q_mid + 0.5 * (q_fin - qf[i, j])

        min_val, max_val = sample_minmax(qf, p_mid)
        cond = min_val < new_qf[i, j] < max_val
        for k in ti.static(range(cond.n)):
            if not cond[k]:
                new_qf[i, j][k] = q_mid[k]


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
    ti.cache_read_only(vf)
    for i, j in vf:
        vl = sample(vf, i - 1, j).x
        vr = sample(vf, i + 1, j).x
        vb = sample(vf, i, j - 1).y
        vt = sample(vf, i, j + 1).y
        vc = sample(vf, i, j)
        if i == 0:
            vl = -vc.x
        if i == res - 1:
            vr = -vc.x
        if j == 0:
            vb = -vc.y
        if j == res - 1:
            vt = -vc.y
        velocity_divs[i, j] = (vr - vl + vt - vb) * 0.5


@ti.kernel
def vorticity(vf: ti.template()):
    ti.cache_read_only(vf)
    for i, j in vf:
        vl = sample(vf, i - 1, j).y
        vr = sample(vf, i + 1, j).y
        vb = sample(vf, i, j - 1).x
        vt = sample(vf, i, j + 1).x
        vc = sample(vf, i, j)
        velocity_curls[i, j] = (vr - vl - vt + vb) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    ti.cache_read_only(pf)
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    ti.cache_read_only(pf)
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    # anti-physics visual enhancement...
    ti.cache_read_only(cf)
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb), abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = min(max(vf[i, j] + force * dt, -1e3), 1e3)


def step(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)

    divergence(velocities_pair.cur)

    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

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
    #gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
    #divergence(velocities_pair.cur); gui.set_image(velocity_divs.to_numpy() * 0.1 + 0.5)
    #vorticity(velocities_pair.cur); gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
    gui.show()
