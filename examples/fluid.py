# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html

import taichi as ti
import numpy as np
import time

res = 600
dx = 1.0
inv_dx = 1.0 / dx
half_inv_dx = 0.5 * inv_dx
dt = 0.03
p_jacobi_iters = 30
f_strength = 10000.0
vel_decay = 1.0
debug = True

assert res > 2

ti.init(arch=ti.cuda)

_velocities = ti.Vector(2, dt=ti.f32, shape=(res, res))
_new_velocities = ti.Vector(2, dt=ti.f32, shape=(res, res))
velocity_divs = ti.var(dt=ti.f32, shape=(res, res))
_pressures = ti.var(dt=ti.f32, shape=(res, res))
_new_pressures = ti.var(dt=ti.f32, shape=(res, res))
color_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)


@ti.func
def sample(qf, u, v):
    i, j = int(u), int(v)
    # Nearest
    i = max(0, min(res - 1, i))
    j = max(0, min(res - 1, j))
    return qf[i, j]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, u, v):
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


@ti.kernel
def advect(vf: ti.template(), new_vf: ti.template()):
    for i, j in vf:
        coord = ti.Vector([i, j]) + 0.5 - dt * vf[i, j]
        new_vf[i, j] = bilerp(vf, coord[0], coord[1]) * vel_decay


inv_force_radius = (3.0 / res)
f_strength_dt = f_strength * dt


@ti.kernel
def apply_mouse_impulse(vf: ti.template(), mouse_arr: ti.ext_arr()):
    for i, j in vf:
        omx, omy = mouse_arr[2], mouse_arr[3]
        mdir = ti.Vector([mouse_arr[0], mouse_arr[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        momentum = mdir * f_strength_dt * ti.exp(-d2 * inv_force_radius)
        v = vf[i, j]
        vf[i, j] = v + momentum


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)[0]
        vr = sample(vf, i + 1, j)[0]
        vb = sample(vf, i, j - 1)[1]
        vt = sample(vf, i, j + 1)[1]
        vc = sample(vf, i, j)
        if i == 0:
            vl = -vc[0]
        if i == res - 1:
            vr = -vc[0]
        if j == 0:
            vb = -vc[1]
        if j == res - 1:
            vt = -vc[1]
        velocity_divs[i, j] = (vr - vl + vt - vb) * half_inv_dx


p_alpha = -dx * dx


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt + p_alpha * div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        v = sample(vf, i, j)
        v = v - half_inv_dx * ti.Vector([pr - pl, pt - pb])
        vf[i, j] = v


@ti.kernel
def fill_color_v2(vf: ti.template()):
    for i, j in vf:
        v = vf[i, j]
        color_buffer[i, j] = ti.Vector([abs(v[0]), abs(v[1]), 0.25])


@ti.kernel
def fill_color_s(sf: ti.template()):
    for i, j in sf:
        s = abs(sf[i, j])
        color_buffer[i, j] = ti.Vector([s, s * 0.25, 0.2])


def step(mouse_data):
    advect(velocities_pair.cur, velocities_pair.nxt)
    velocities_pair.swap()

    apply_mouse_impulse(velocities_pair.cur, mouse_data)

    divergence(velocities_pair.cur)
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)
    fill_color_s(velocity_divs)
    # fill_color_v2(velocities_pair.cur)

    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f'divergence={div_s}')


def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)


class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        mouse_data = np.array([0] * 4, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = vec2_npf32(gui.get_cursor_pos()) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
        return mouse_data


def reset():
    velocities_pair.cur.fill(ti.Vector([0, 0]))
    pressures_pair.cur.fill(0.0)
    color_buffer.fill(ti.Vector([0, 0, 0]))


def main():
    global debug
    gui = ti.GUI('Fluid', (res, res))
    md_gen = MouseDataGen()
    paused = False
    while True:
        while gui.has_key_event():
            e = gui.get_key_event()
            if e.type == ti.GUI.RELEASE:
                continue
            elif e.key == ti.GUI.ESCAPE:
                exit(0)
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

        img = color_buffer.to_numpy(as_vector=True)
        gui.set_image(img)
        gui.show()


if __name__ == '__main__':
    main()
