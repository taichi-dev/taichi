# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import argparse

import numpy as np

import taichi as ti
from taichi.math import vec2, vec3

ti.init(arch=ti.cuda)

res = 512
dt = 0.03
p_jacobi_iters = 500  # 40 for a quicker but less accurate result
f_strength = 10000.0
curl_strength = 0
time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
force_radius = res / 2.0


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


@ti.func
def sample_impl(qf: ti.template(), u: int, v: int):
    I = ti.Vector([u, v])
    I = ti.max(0, ti.min(res - 1, I))
    return qf[I]


@ti.real_func
def sample2(qf: ti.types.ndarray(ndim=2), u: int, v: int) -> vec2:
    return sample_impl(qf, u, v)


@ti.real_func
def sample3(qf: ti.types.ndarray(ndim=2), u: int, v: int) -> vec3:
    return sample_impl(qf, u, v)


@ti.real_func
def sample0(qf: ti.types.ndarray(ndim=2), u: int, v: int) -> float:
    return sample_impl(qf, u, v)


@ti.func
def sample(qf: ti.template(), u, v):
    if ti.static(qf.element_shape() == [2]):
        return sample2(qf, u, v)
    if ti.static(qf.element_shape() == [3]):
        return sample3(qf, u, v)
    ti.static_assert(qf.element_shape() == [])
    return sample0(qf, u, v)


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp_impl(vf: ti.template(), p):
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


@ti.real_func
def bilerp2(vf: ti.types.ndarray(ndim=2), p: vec2) -> vec2:
    return bilerp_impl(vf, p)


@ti.real_func
def bilerp3(vf: ti.types.ndarray(ndim=2), p: vec2) -> vec3:
    return bilerp_impl(vf, p)


@ti.func
def bilerp(vf: ti.template(), p):
    if ti.static(vf.element_shape() == [2]):
        return bilerp2(vf, p)
    ti.static_assert(vf.element_shape() == [3])
    return bilerp3(vf, p)


# 3rd order Runge-Kutta
@ti.real_func
def backtrace(vf: ti.types.ndarray(ndim=2), p: vec2, dt_: float) -> vec2:
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt_ * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt_ * v2
    v3 = bilerp(vf, p2)
    return p - dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)


@ti.kernel
def advect(
    vf: ti.types.ndarray(ndim=2),
    qf: ti.types.ndarray(ndim=2),
    new_qf: ti.types.ndarray(ndim=2),
):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p) * dye_decay


@ti.kernel
def apply_impulse(
    vf: ti.types.ndarray(ndim=2),
    dyef: ti.types.ndarray(ndim=2),
    imp_data: ti.types.ndarray(ndim=1),
):
    g_dir = -ti.Vector([0, 9.8]) * 300
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)

        dc = dyef[i, j]
        a = dc.norm()

        momentum = (mdir * f_strength * factor + g_dir * a / (1 + a)) * dt

        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15) ** 2)) * ti.Vector([imp_data[4], imp_data[5], imp_data[6]])

        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.types.ndarray(ndim=2), velocity_divs: ti.types.ndarray(ndim=2)):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5


@ti.kernel
def pressure_jacobi(
    pf: ti.types.ndarray(ndim=2),
    new_pf: ti.types.ndarray(ndim=2),
    velocity_divs: ti.types.ndarray(ndim=2),
):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.types.ndarray(ndim=2), pf: ti.types.ndarray(ndim=2)):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt, _velocity_divs)
        pressures_pair.swap()


def step_orig(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)

    divergence(velocities_pair.cur, _velocity_divs)

    solve_pressure_jacobi()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)


mouse_data_ti = ti.ndarray(ti.f32, shape=(8,))


class MouseDataGen:
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
        mouse_data_ti.from_numpy(mouse_data)  # false+, pylint: disable=no-member
        return mouse_data_ti


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)


def main():
    global velocities_pair, pressures_pair, dyes_pair, curl_strength

    paused = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args, _ = parser.parse_known_args()

    gui = ti.GUI("Stable Fluid", (res, res))
    md_gen = MouseDataGen()

    _velocities = ti.Vector.ndarray(2, float, shape=(res, res))
    _new_velocities = ti.Vector.ndarray(2, float, shape=(res, res))
    _velocity_divs = ti.ndarray(float, shape=(res, res))
    _pressures = ti.ndarray(float, shape=(res, res))
    _new_pressures = ti.ndarray(float, shape=(res, res))
    _dye_buffer = ti.Vector.ndarray(3, float, shape=(res, res))
    _new_dye_buffer = ti.Vector.ndarray(3, float, shape=(res, res))

    if args.baseline:
        velocities_pair = TexPair(_velocities, _new_velocities)
        pressures_pair = TexPair(_pressures, _new_pressures)
        dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)
    else:
        print("running in graph mode")
        velocities_pair_cur = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "velocities_pair_cur", dtype=ti.math.vec2, ndim=2)
        velocities_pair_nxt = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "velocities_pair_nxt", dtype=ti.math.vec2, ndim=2)
        dyes_pair_cur = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dyes_pair_cur", dtype=ti.math.vec3, ndim=2)
        dyes_pair_nxt = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dyes_pair_nxt", dtype=ti.math.vec3, ndim=2)
        pressures_pair_cur = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pressures_pair_cur", dtype=ti.f32, ndim=2)
        pressures_pair_nxt = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pressures_pair_nxt", dtype=ti.f32, ndim=2)
        velocity_divs = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "velocity_divs", dtype=ti.f32, ndim=2)
        mouse_data = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "mouse_data", dtype=ti.f32, ndim=1)

        g1_builder = ti.graph.GraphBuilder()
        g1_builder.dispatch(advect, velocities_pair_cur, velocities_pair_cur, velocities_pair_nxt)
        g1_builder.dispatch(advect, velocities_pair_cur, dyes_pair_cur, dyes_pair_nxt)
        g1_builder.dispatch(apply_impulse, velocities_pair_nxt, dyes_pair_nxt, mouse_data)
        g1_builder.dispatch(divergence, velocities_pair_nxt, velocity_divs)
        # swap is unrolled in the loop so we only need p_jacobi_iters // 2 iterations.
        for _ in range(p_jacobi_iters // 2):
            g1_builder.dispatch(pressure_jacobi, pressures_pair_cur, pressures_pair_nxt, velocity_divs)
            g1_builder.dispatch(pressure_jacobi, pressures_pair_nxt, pressures_pair_cur, velocity_divs)
        g1_builder.dispatch(subtract_gradient, velocities_pair_nxt, pressures_pair_cur)
        g1 = g1_builder.compile()

        g2_builder = ti.graph.GraphBuilder()
        g2_builder.dispatch(advect, velocities_pair_nxt, velocities_pair_nxt, velocities_pair_cur)
        g2_builder.dispatch(advect, velocities_pair_nxt, dyes_pair_nxt, dyes_pair_cur)
        g2_builder.dispatch(apply_impulse, velocities_pair_cur, dyes_pair_cur, mouse_data)
        g2_builder.dispatch(divergence, velocities_pair_cur, velocity_divs)
        for _ in range(p_jacobi_iters // 2):
            g2_builder.dispatch(pressure_jacobi, pressures_pair_cur, pressures_pair_nxt, velocity_divs)
            g2_builder.dispatch(pressure_jacobi, pressures_pair_nxt, pressures_pair_cur, velocity_divs)
        g2_builder.dispatch(subtract_gradient, velocities_pair_cur, pressures_pair_cur)
        g2 = g2_builder.compile()

    swap = True

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == "r":
                paused = False
                reset()
            elif e.key == "s":
                if curl_strength:
                    curl_strength = 0
                else:
                    curl_strength = 7
            elif e.key == "p":
                paused = not paused

        if not paused:
            _mouse_data = md_gen(gui)
            if args.baseline:
                step_orig(_mouse_data)
                gui.set_image(dyes_pair.cur.to_numpy())
            else:
                invoke_args = {
                    "mouse_data": _mouse_data,
                    "velocities_pair_cur": _velocities,
                    "velocities_pair_nxt": _new_velocities,
                    "dyes_pair_cur": _dye_buffer,
                    "dyes_pair_nxt": _new_dye_buffer,
                    "pressures_pair_cur": _pressures,
                    "pressures_pair_nxt": _new_pressures,
                    "velocity_divs": _velocity_divs,
                }
                if swap:
                    g1.run(invoke_args)
                    gui.set_image(_dye_buffer.to_numpy())  # false+, pylint: disable=no-member
                    swap = False
                else:
                    g2.run(invoke_args)
                    gui.set_image(_new_dye_buffer.to_numpy())  # false+, pylint: disable=no-member
                    swap = True
        gui.show()


if __name__ == "__main__":
    main()
