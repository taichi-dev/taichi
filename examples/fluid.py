# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html

import taichi as ti
import numpy as np

res = 600
half_res = res * 0.5
dx = 1.0
rdx = 1.0 / dx
half_rdx = 0.5 * rdx
dt = 0.025
v_decay = 0.95
p_jacobi_iters = 30
f_strength = 10000.0

enable_diffusion = False
viscority = 10.0
v_jacobi_iters = 20

ti.init(arch=ti.cuda)

velocities = ti.Vector(2, dt=ti.f32, shape=(res, res))
new_velocities = ti.Vector(2, dt=ti.f32, shape=(res, res))
div_vels = ti.var(dt=ti.f32, shape=(res, res))
pressures = ti.var(dt=ti.f32, shape=(res, res))
new_pressures = ti.var(dt=ti.f32, shape=(res, res))
color_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))


@ti.func
def in_boundary(i, j):
    return 0 <= i and i < res and 0 <= j and j < res


@ti.func
def sample_v(vf, x, y):
    i, j = int(x), int(y)
    # implicitly enforces the boundary condition: v(boundary+) = 0
    v = ti.Vector([0.0, 0.0])
    if in_boundary(i, j):
        v = vf[i, j]
    return v


@ti.func
def sample_p(pf, i, j):
    # implicitly enforces the boundary condition: p(boundary+) = p(boundary)
    i = ti.max(0, ti.min(i, res - 1))
    j = ti.max(0, ti.min(j, res - 1))
    return pf[i, j]


@ti.func
def lerp(l, r, vl, vr, x):
    return vl + (vr - vl) * (x - l) / (r - l)


@ti.kernel
def advect(vf: ti.template(), new_vf: ti.template()):
    for i, j in vf:
        # q(x, t + dt) = q(x - u(x, t) * dt, t)
        pos = ti.Vector([i, j]) + 0.5 - dt * vf[i, j]
        from_i, from_j = int(pos[0]), int(pos[1])
        if not in_boundary(from_i, from_j):
            new_vf[i, j] = ti.Vector([0.0, 0.0])
            continue
        # bilinear interpolation
        cx1, cy1 = from_i + 0.5, from_j + 0.5
        cx2, cy2 = cx1 + 1.0, cy1 + 1.0
        if pos[0] < cx1:
            cx2 = cx1 - 1.0
        if pos[1] < cy1:
            cy2 = cy1 - 1.0
        v_y1 = lerp(cx1, cx2, sample_v(vf, cx1, cy1), sample_v(vf, cx2, cy1),
                    pos[0])
        v_y2 = lerp(cx1, cx2, sample_v(vf, cx1, cy2), sample_v(vf, cx2, cy2),
                    pos[0])
        new_v = lerp(cy1, cy2, v_y1, v_y2, pos[1])
        new_vf[i, j] = new_v


v_alpha = dx * dx / (viscority * dt)
v_rbeta = 1.0 / (4.0 + v_alpha)


@ti.kernel
def diffusion_jacobi(vf: ti.template(), new_vf: ti.template()):
    for i, j in vf:
        vl = sample_v(vf, i - 1, j)
        vr = sample_v(vf, i + 1, j)
        vt = sample_v(vf, i, j + 1)
        vb = sample_v(vf, i, j - 1)

        b = vf[i, j]
        new_vf[i, j] = (vl + vr + vt + vb + b * v_alpha) * v_rbeta


rforce_radius = (3.0 / res)
f_strength_dt = f_strength * dt


@ti.kernel
def apply_mouse_impulse(new_vf: ti.template(), mouse_arr: ti.ext_arr()):
    for i, j in new_vf:
        omx, omy = mouse_arr[2], mouse_arr[3]
        mdir = ti.Vector([mouse_arr[0], mouse_arr[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        momentum = mdir * f_strength_dt * ti.exp(-d2 * rforce_radius)
        v = new_vf[i, j]
        v += momentum
        new_vf[i, j] = v


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample_v(vf, i - 1, j)[0]
        vr = sample_v(vf, i + 1, j)[0]
        vt = sample_v(vf, i, j + 1)[1]
        vb = sample_v(vf, i, j - 1)[1]
        div_vels[i, j] = ((vr - vl) + (vt - vb)) * half_rdx


p_alpha = -dx * dx


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample_p(pf, i - 1, j)
        pr = sample_p(pf, i + 1, j)
        pt = sample_p(pf, i, j + 1)
        pb = sample_p(pf, i, j - 1)
        div = div_vels[i, j]
        new_pf[i, j] = (pl + pr + pt + pb + p_alpha * div) * 0.25


@ti.kernel
def subtract_gradient(new_vf: ti.template(), pf: ti.template()):
    # Make |new_vf| divergence free
    for i, j in new_vf:
        pl = sample_p(pf, i - 1, j)
        pr = sample_p(pf, i + 1, j)
        pt = sample_p(pf, i, j + 1)
        pb = sample_p(pf, i, j - 1)
        v = new_vf[i, j]
        v -= half_rdx * ti.Vector([pr - pl, pt - pb])
        v *= v_decay
        new_vf[i, j] = v


@ti.kernel
def fill_color(new_vf: ti.template()):
    for i, j in new_vf:
        v = new_vf[i, j]
        color_buffer[i, j] = ti.Vector([ti.abs(v[0]), ti.abs(v[1]), 0.25])


def reset():
    velocities.fill(ti.Vector([0.0, 0.0]))
    pressures.fill(0.0)
    color_buffer.fill(ti.Vector([0.0, 0.0, 0.0]))


def step(mouse_data):
    global velocities, new_velocities, pressures, new_pressures
    advect(velocities, new_velocities)
    if enable_diffusion:
        for _ in range(v_jacobi_iters):
            diffusion_jacobi(velocities, new_velocities)
            velocities, new_velocities = new_velocities, velocities
    apply_mouse_impulse(new_velocities, mouse_data)
    divergence(new_velocities)
    pressures.fill(0.0)
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures, new_pressures)
        pressures, new_pressures = new_pressures, pressures
    subtract_gradient(new_velocities, pressures)
    fill_color(new_velocities)
    velocities, new_velocities = new_velocities, velocities


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


def main():
    gui = ti.GUI('Fluid', (res, res))
    md_gen = MouseDataGen()
    while True:
        while gui.has_key_event():
            e = gui.get_key_event()
            if e.type == ti.GUI.RELEASE:
                continue
            elif e.key == ti.GUI.ESCAPE:
                exit(0)
            elif e.key == 'r':
                reset()

        mouse_data = md_gen(gui)
        step(mouse_data)

        img = color_buffer.to_numpy(as_vector=True)
        gui.set_image(img)
        gui.show()


if __name__ == '__main__':
    main()
