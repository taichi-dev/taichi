# C++ reference and tutorial (Chinese): https://zhuanlan.zhihu.com/p/26882619
import math

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

eps = 0.01
dt = 0.1

n_vortex = 4
n_tracer = 200000

pos = ti.Vector.field(2, ti.f32, shape=n_vortex)
new_pos = ti.Vector.field(2, ti.f32, shape=n_vortex)
vort = ti.field(ti.f32, shape=n_vortex)

tracer = ti.Vector.field(2, ti.f32, shape=n_tracer)


@ti.func
def compute_u_single(p, i):
    r2 = (p - pos[i]).norm() ** 2
    uv = ti.Vector([pos[i].y - p.y, p.x - pos[i].x])
    return vort[i] * uv / (r2 * math.pi) * 0.5 * (1.0 - ti.exp(-r2 / eps**2))


@ti.func
def compute_u_full(p):
    u = ti.Vector([0.0, 0.0])
    for i in range(n_vortex):
        u += compute_u_single(p, i)
    return u


@ti.kernel
def integrate_vortex():
    for i in range(n_vortex):
        v = ti.Vector([0.0, 0.0])
        for j in range(n_vortex):
            if i != j:
                v += compute_u_single(pos[i], j)
        new_pos[i] = pos[i] + dt * v

    for i in range(n_vortex):
        pos[i] = new_pos[i]


@ti.kernel
def advect():
    for i in range(n_tracer):
        # Ralston's third-order method
        p = tracer[i]
        v1 = compute_u_full(p)
        v2 = compute_u_full(p + v1 * dt * 0.5)
        v3 = compute_u_full(p + v2 * dt * 0.75)
        tracer[i] += (2 / 9 * v1 + 1 / 3 * v2 + 4 / 9 * v3) * dt


pos[0] = [0, 1]
pos[1] = [0, -1]
pos[2] = [0, 0.3]
pos[3] = [0, -0.3]
vort[0] = 1
vort[1] = -1
vort[2] = 1
vort[3] = -1


@ti.kernel
def init_tracers():
    for i in range(n_tracer):
        tracer[i] = [ti.random() - 0.5, ti.random() * 3 - 1.5]


def main():
    init_tracers()
    gui = ti.GUI("Vortex Rings", (1024, 512), background_color=0xFFFFFF)

    while gui.running:
        for i in range(4):  # substeps
            advect()
            integrate_vortex()

        gui.circles(
            tracer.to_numpy() * np.array([[0.05, 0.1]]) + np.array([[0.0, 0.5]]),
            radius=0.5,
            color=0x0,
        )

        gui.show()


if __name__ == "__main__":
    main()
