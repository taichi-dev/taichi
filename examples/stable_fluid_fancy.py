# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation

import taichi as ti
import os
import numpy as np
import time
from contextlib import contextmanager


# SIM_RES_x = 139
# SIM_RES_y = 199

SIM_RES_x = 417
SIM_RES_y = 597

RENDER_RES_x = 1668
RENDER_RES_y = 2388

# RENDER_RES_x = 417
# RENDER_RES_y = 597

dt = 0.01
p_jacobi_iters = 400
debug = False

# assert res > 2

ti.init(arch=ti.metal)


@ti.func
def sample_clamp_to_edge(qf, u, v, res):
    resX, resY = res
    i, j = int(u), int(v)
    # clamp to edge
    i = max(0, min(resX - 1, i))
    j = max(0, min(resY - 1, j))
    return qf[i, j]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, u, v, res):
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = int(s), int(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample_clamp_to_edge(vf, iu, iv, res)
    b = sample_clamp_to_edge(vf, iu + 1, iv, res)
    c = sample_clamp_to_edge(vf, iu, iv + 1, res)
    d = sample_clamp_to_edge(vf, iu + 1, iv + 1, res)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

@ti.func 
def sample(qf, uv, res):
    resx, resy = res
    u, v = uv
    u *= resx
    v *= resy
    return bilerp(qf, u, v, res)

@ti.func
def normalize(ij, res):
    i, j = ij
    texel_sizeX = 1.0 / res.x
    texel_sizeY = 1.0 / res.y
    u = i * texel_sizeX
    v = j * texel_sizeY
    return ti.Vector([u, v])

_dye_buffer = ti.Vector.field(4, ti.f32, shape=(RENDER_RES_x, RENDER_RES_y))
_new_dye_buffer = ti.Vector.field(4, ti.f32, shape=(RENDER_RES_x, RENDER_RES_y))
_velocities = ti.Vector.field(2, ti.f32, shape=(SIM_RES_x, SIM_RES_y))
_new_velocities = ti.Vector.field(2, ti.f32, shape=(SIM_RES_x, SIM_RES_y))

velocity_divs = ti.field(float, shape=(SIM_RES_x, SIM_RES_y))
_curls = ti.field(float, shape=(SIM_RES_x, SIM_RES_y))
_pressures = ti.field(float, shape=(SIM_RES_x, SIM_RES_y))
_new_pressures = ti.field(float, shape=(SIM_RES_x, SIM_RES_y))

# color_buffer = Texture.Vector(3, RENDER_RES)


# def make_bloom_mipmap():
#     cur_res = SIM_RES
#     mm = []
#     BLOOM_ITERS = 8
#     # while cur_res > 2:
#     for _ in range(BLOOM_ITERS):
#         cur_res = (cur_res >> 1)
#         if cur_res < 4:
#             break
#         mm.append(Texture.Vector(3, cur_res))
#     return mm


# _bloom_final = Texture.Vector(3, SIM_RES)
# _bloom_mipmap = make_bloom_mipmap()

# _sunrays = Texture.Scalar(SIM_RES)
# _sunrays_scratch = Texture.Scalar(SIM_RES)


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


@ti.kernel

def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template(),
           dissipation: float, resX: ti.i32, resY: ti.i32):
    for i, j in qf:
        res = ti.Vector([resX, resY])
        uv = normalize(ti.Vector([i, j]) + 0.5, res)
        sim_res = ti.Vector([SIM_RES_x, SIM_RES_y])
        vel = sample(vf, uv, sim_res)
        vel[1] /= SIM_RES_y / SIM_RES_x
        prev_uv = uv - dt * vel
        q_s = sample(qf, prev_uv, res)
        decay = 1.0 + dissipation * dt
        new_qf[i, j] = q_s / decay


force_radius = 0.1 / 100
inv_force_radius = 1.0 / force_radius


@ti.kernel
def apply_impulse(dye: ti.template(),
        vf: ti.template(), omx: float, omy: float, fx: float, fy: float, 
        r: float, g: float, b: float):
    for i, j in vf:
        sim_res = ti.Vector([SIM_RES_x, SIM_RES_y])
        u, v = normalize(ti.Vector([i, j]) + 0.5, sim_res)
        dx, dy = (u - omx), (v*SIM_RES_y / SIM_RES_x - omy)
        d2 = dx * dx + dy * dy
        momentum = ti.exp(-d2 * inv_force_radius) * ti.Vector([fx, fy])
        vel = vf[i, j]
        vf[i, j] = vel + momentum
    for i, j in dye:
        render_res = ti.Vector([RENDER_RES_x, RENDER_RES_y])
        u, v = normalize(ti.Vector([i, j]) + 0.5, render_res)
        dx, dy = (u - omx), (v*SIM_RES_y / SIM_RES_x - omy)
        d2 = dx * dx + dy * dy
        impulse = ti.exp(-d2 * inv_dye_radius) * ti.Vector([r, g, b, 1.0])
        col = dye[i, j]
        col += impulse
        dye[i, j] = col

dye_radius = 0.1 / 100
inv_dye_radius = 1.0 / dye_radius


@ti.kernel
def add_curl(vf: ti.template()):
    for i, j in _curls:
        res = ti.Vector([SIM_RES_x, SIM_RES_y])
        vl = sample_clamp_to_edge(vf, i - 1, j, res)[1]
        vr = sample_clamp_to_edge(vf, i + 1, j, res)[1]
        vb = sample_clamp_to_edge(vf, i, j - 1, res)[0]
        vt = sample_clamp_to_edge(vf, i, j + 1, res)[0]
        vort = vr - vl - vt + vb
        _curls[i, j] = 0.5 * vort


curl_strength = 30.0


@ti.kernel
def add_voriticity(vf: ti.template()):
    for i, j in vf:
        res = ti.Vector([SIM_RES_x, SIM_RES_y])
        vl = sample_clamp_to_edge(_curls, i - 1, j, res)
        vr = sample_clamp_to_edge(_curls, i + 1, j, res)
        vb = sample_clamp_to_edge(_curls, i, j - 1, res)
        vt = sample_clamp_to_edge(_curls, i, j + 1, res)
        vc = sample_clamp_to_edge(_curls, i, j, res)

        force = 0.5 * ti.Vector([abs(vt) - abs(vb),
                                 abs(vr) - abs(vl)]).normalized(1e-3)
        force *= curl_strength * vc
        vel = vf[i, j]
        vf[i, j] = min(max(vel + force * dt, -1e3), 1e3)


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        res = ti.Vector([SIM_RES_x, SIM_RES_y])
        vl = sample_clamp_to_edge(vf, i - 1, j, res)[0]
        vr = sample_clamp_to_edge(vf, i + 1, j, res)[0]
        vb = sample_clamp_to_edge(vf, i, j - 1, res)[1]
        vt = sample_clamp_to_edge(vf, i, j + 1, res)[1]
        vc = sample_clamp_to_edge(vf, i, j, res)
        if i == 0:
            vl = -vc[0]
        if i == res.x - 1:
            vr = -vc[0]
        if j == 0:
            vb = -vc[1]
        if j == res.y - 1:
            vt = -vc[1]
        velocity_divs[i, j] = 0.5 * res.y * (vr - vl + vt - vb)
        

p_alpha = - (1.0 / SIM_RES_y) * (1.0 / SIM_RES_y)


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        res = ti.Vector([SIM_RES_x, SIM_RES_y])
        pl = sample_clamp_to_edge(pf, i - 1, j, res)
        pr = sample_clamp_to_edge(pf, i + 1, j, res)
        pb = sample_clamp_to_edge(pf, i, j - 1, res)
        pt = sample_clamp_to_edge(pf, i, j + 1, res)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt + p_alpha * div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        res = ti.Vector([SIM_RES_x, SIM_RES_y])
        pl = sample_clamp_to_edge(pf, i - 1, j, res)
        pr = sample_clamp_to_edge(pf, i + 1, j, res)
        pb = sample_clamp_to_edge(pf, i, j - 1, res)
        pt = sample_clamp_to_edge(pf, i, j + 1, res)
        vel = sample_clamp_to_edge(vf, i, j, res)
        vel -= 0.5 * res * ti.Vector([pr - pl, pt - pb])
        vf[i, j] = vel


filename = os.path.basename(__file__)[:-3]
m = ti.aot.Module(ti.metal)
m.add_field("_dye_buffer", _dye_buffer)
m.add_field("_new_dye_buffer", _new_dye_buffer)
m.add_field("_velocities", _velocities)
m.add_field("_new_velocties", _new_velocities)
m.add_field("_velocity_divs", velocity_divs)
# m.add_field("_curls", _curls)
m.add_field("_pressures", _pressures)
m.add_field("_new_pressures", _new_pressures)

with m.add_kernel_template(advect) as kt:
    kt.instantiate(vf = velocities_pair.cur, qf = velocities_pair.cur, new_qf = velocities_pair.nxt)
    kt.instantiate(vf = velocities_pair.nxt, qf = velocities_pair.nxt, new_qf = velocities_pair.cur)
    kt.instantiate(vf = velocities_pair.cur, qf = dyes_pair.cur, new_qf = dyes_pair.nxt)
    kt.instantiate(vf = velocities_pair.nxt, qf = dyes_pair.nxt, new_qf = dyes_pair.cur)

with m.add_kernel_template(apply_impulse) as kt2:
    kt2.instantiate(dye = dyes_pair.cur, vf = velocities_pair.cur)
    kt2.instantiate(dye = dyes_pair.nxt, vf = velocities_pair.nxt)

with m.add_kernel_template(divergence) as kt:
    kt.instantiate(vf = velocities_pair.cur)
    kt.instantiate(vf = velocities_pair.nxt)

with m.add_kernel_template(pressure_jacobi) as kt:
    kt.instantiate(pf = pressures_pair.cur, new_pf = pressures_pair.nxt)
    kt.instantiate(pf = pressures_pair.nxt, new_pf = pressures_pair.cur)

with m.add_kernel_template(subtract_gradient) as kt:
    kt.instantiate(vf = velocities_pair.cur, pf = pressures_pair.cur)
    kt.instantiate(vf = velocities_pair.nxt, pf = pressures_pair.cur)

m.save('outputaot', filename)


# @ti.kernel
# def fill_color_v2(vf: ti.template()):
#     for i, j in vf:
#         v = vf.field[i, j]
#         color_buffer.field[i, j] = ti.Vector([abs(v[0]), abs(v[1]), 0.25])


# @ti.func
# def linear_to_gamma(rgb):
#     rgb = max(rgb, 0)
#     EXP = 0.416666667
#     return max(1.055 * ti.pow(rgb, EXP) - 0.055, 0)


# @ti.kernel
# def fill_color_v3(dye: ti.template()):
#     for i, j in color_buffer.field:
#         uv = dye.normalize(ti.Vector([i, j]) + 0.5)
#         v = dye.sample(uv)
#         c = ti.Vector([abs(v[0]), abs(v[1]), abs(v[2])])

#         sunrays = _sunrays.sample(uv)
#         c *= sunrays

#         bloom = _bloom_final.sample(uv) * 0.25
#         bloom *= sunrays
#         bloom = linear_to_gamma(bloom)
#         c += bloom

#         color_buffer.field[i, j] = c
    
# BLOOM_THRESHOLD = 0.6
# BLOOM_SOFT_KNEE = 0.7
# BLOOM_KNEE = BLOOM_THRESHOLD * BLOOM_SOFT_KNEE + 0.0001
# BLOOM_CURVE_X = BLOOM_THRESHOLD - BLOOM_KNEE
# BLOOM_CURVE_Y = BLOOM_KNEE * 2
# BLOOM_CURVE_Z = 0.25 / BLOOM_KNEE


# @ti.kernel
# def bloom_prefilter(qf: ti.template()):
#     # assuming qf is a dye field
#     for i, j in _bloom_final.field:
#         uv = _bloom_final.normalize(ti.Vector([i, j]) + 0.5)
#         # vi, vj = int(i * ratio), int(j * ratio)
#         # coord = ti.Vector([i, j]) + 0.5 - dt * vf[vi, vj] / ratio
#         c = qf.sample(uv)
#         br = max(c[0], max(c[1], c[2]))
#         rq = min(max(br - BLOOM_CURVE_X, 0), BLOOM_CURVE_Y)
#         rq = BLOOM_CURVE_Z * rq * rq
#         c *= max(rq, br - BLOOM_THRESHOLD) / max(br, 0.0001)
#         _bloom_final.field[i, j] = c


# @ti.kernel
# def bloom_fwd_blur(src: ti.template(), dst: ti.template()):
#     for i, j in dst.field:
#         u, v = dst.normalize(ti.Vector([i, j]) + 0.5)
#         texel_sz = dst.texel_size
#         c = ti.Vector([0.0, 0.0, 0.0])
#         c += src.sample_sep(u - texel_sz, v)
#         c += src.sample_sep(u + texel_sz, v)
#         c += src.sample_sep(u, v - texel_sz)
#         c += src.sample_sep(u, v + texel_sz)
#         # c = src.sample(v)
#         dst.field[i, j] = c * 0.25


# @ti.kernel
# def bloom_inv_blur(src: ti.template(), dst: ti.template()):
#     for i, j in dst.field:
#         u, v = dst.normalize(ti.Vector([i, j]) + 0.5)
#         texel_sz = dst.texel_size
#         c = ti.Vector([0.0, 0.0, 0.0])
#         c += src.sample_sep(u - texel_sz, v)
#         c += src.sample_sep(u + texel_sz, v)
#         c += src.sample_sep(u, v - texel_sz)
#         c += src.sample_sep(u, v + texel_sz)
#         dst.field[i, j] += c * 0.25


# def apply_bloom(qf):
#     bloom_prefilter(qf)
#     last = _bloom_final
#     for bm in _bloom_mipmap:
#         bloom_fwd_blur(last, bm)
#         last = bm
#     for i in reversed(range(len(_bloom_mipmap) - 1)):
#         bm = _bloom_mipmap[i]
#         bloom_inv_blur(last, bm)
#         last = bm
#     bloom_inv_blur(last, _bloom_final)


# @ti.kernel
# def k_sunrays_mask(dye_r: ti.template()):
#     for i, j in _sunrays_scratch.field:
#         uv = _sunrays_scratch.normalize(ti.Vector([i, j]) + 0.5)
#         r, g, b = dye_r.sample(uv)
#         br = max(r, max(g, b))
#         a = 1.0 - min(max(br * 20.0, 0), 0.8)
#         _sunrays_scratch.field[i, j] = a


# SUNRAYS_DENSITY = 0.3
# SUNRAYS_DECAY = 0.95
# SUNRAYS_EXPOSURE = 0.7
# SUNRAYS_ITERATIONS = 16


# @ti.kernel
# def k_sunrays():
#     for i, j in _sunrays.field:
#         cur_coord = _sunrays_scratch.normalize(ti.Vector([i, j]) + 0.5)
#         dir = cur_coord - 0.5
#         dir *= (SUNRAYS_DENSITY / SUNRAYS_ITERATIONS)
#         illumination_decay = 1.0
#         total_color = _sunrays_scratch.field[i, j]
#         for _ in range(SUNRAYS_ITERATIONS):
#             cur_coord -= dir
#             col = _sunrays_scratch.sample(cur_coord)
#             total_color += col * illumination_decay
#             illumination_decay *= SUNRAYS_DECAY
#         _sunrays.field[i, j] = total_color * SUNRAYS_EXPOSURE


# def apply_sunrays(qf):
#     k_sunrays_mask(qf)
#     k_sunrays()


# def step(mouse_data):
#     advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt, 0.8, SIM_RES_x, SIM_RES_y)
#     advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt, 0.8, RENDER_RES_x, RENDER_RES_y)
#     velocities_pair.swap()
#     dyes_pair.swap()

#     f_strength = 6000.0

#     normed_mxy = mouse_data[2:4]
#     rgb = mouse_data[4:]
#     force = mouse_data[0:2] * f_strength * dt
#     apply_impulse(dyes_pair.cur, velocities_pair.cur, float(normed_mxy[0]), float(normed_mxy[1]),
#                      float(force[0]), float(force[1]),
#                      float(rgb[0]), float(rgb[1]), float(rgb[2]))


#     # add_curl(velocities_pair.cur)
#     # add_voriticity(velocities_pair.cur)
#     # velocities_pair.swap()

#     divergence(velocities_pair.cur)
#     for _ in range(p_jacobi_iters):
#         pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
#         pressures_pair.swap()

#     subtract_gradient(velocities_pair.cur, pressures_pair.cur)
#     # apply_bloom(dyes_pair.cur)
#     # apply_sunrays(dyes_pair.cur)
#     # fill_color_v3(dyes_pair.cur)


# def vec2_npf32(m):
#     return np.array([m[0], m[1]], dtype=np.float32)


# def hsv_to_rgb(h, s, v):
#     i = int(h * 6.0)
#     f = h * 6.0 - i
#     p = v * (1 - s)
#     q = v * (1 - f * s)
#     t = v * (1 - (1 - f) * s)
#     colors = [
#         (v, t, p),
#         (q, v, p),
#         (p, v, t),
#         (p, q, v),
#         (t, p, v),
#         (v, p, q),
#     ]
#     return colors[i]


# class MouseDataGen(object):
#     def __init__(self):
#         self.prev_mouse = None
#         self.prev_color = None

#     def __call__(self, gui):
#         # [0:2]: delta direction  (not normalized)
#         # [2:4]: current mouse xy (normalized)
#         # [4:7]: color
#         mouse_data = np.array([0] * 8, dtype=np.float32)
#         if gui.is_pressed(ti.GUI.LMB):
#             mxy = vec2_npf32(gui.get_cursor_pos())
#             mxy[1] *= SIM_RES_y / SIM_RES_x
#             if self.prev_mouse is None:
#                 self.prev_mouse = mxy
#                 # Set lower bound to 0.3 to prevent too dark colors
#                 self.prev_color = (np.random.rand(3) * 0.7) + 0.3
#                 # sat = np.random.random() * 0.4 + 0.6
#                 # self.prev_color = hsv_to_rgb(np.random.random(), sat, 1)
#             else:
#                 mdir = mxy - self.prev_mouse
#                 # mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
#                 mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
#                 mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
#                 mouse_data[4:7] = self.prev_color
#                 self.prev_mouse = mxy
#         else:
#             self.prev_mouse = None
#             self.prev_color = None
#         return mouse_data


# def reset():
#     velocities_pair.cur.fill(ti.Vector([0, 0]))
#     pressures_pair.cur.fill(0.0)
#     dyes_pair.cur.fill(ti.Vector([0, 0, 0]))
#     # color_buffer.fill(ti.Vector([0, 0, 0]))


# def main():
#     global debug
#     gui = ti.GUI('Stable-Fluid', (RENDER_RES_x, RENDER_RES_y))
#     md_gen = MouseDataGen()
#     paused = False
#     while True:
#         while gui.get_event(ti.GUI.PRESS):
#             e = gui.event
#             if e.key == ti.GUI.ESCAPE:
#                 exit(0)
#             elif e.key == 'r':
#                 paused = False
#                 reset()
#             elif e.key == 'p':
#                 paused = not paused
#             elif e.key == 'd':
#                 debug = not debug

#         if not paused:
#             mouse_data = md_gen(gui)
#             step(mouse_data)

#         img = _dye_buffer.to_numpy()
#         gui.set_image(img)
#         gui.show()


# if __name__ == '__main__':
#     main()