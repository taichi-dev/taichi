# Authored by Luhuai Jiao
# This is a 1D simulation of "two-stream instability" in Plasma Physicis.
# Some settings of the grids and particles are taken from "Introduction to Computational Plasma Physics"(ISBN: 9787030563675)

import taichi as ti

ti.init(arch=ti.gpu)  # Try to run on GPU
PI = 3.141592653589793
L = 8 * PI  # simulation domain size
dt = 0.1  # time step
substepping = 8
ng = 32  # number of grids
np = 16384  # numer of particles
vb = 1.0  # beam-velocity, one is vb, the other is -vb
vt = 0.3  # thermal velocity
wp = 1  # Plasma frequence
qm = -1  # charge-mass ratio
q = wp * wp / (qm * np / L)  # charge of a particle
rho_back = -q * np / L  # background charge density
dx = L / ng  # grid spacing
inv_dx = 1.0 / dx
x = ti.Vector.field(1, ti.f32, np)  # position
v = ti.Vector.field(1, ti.f32, np)  # velocity
rho = ti.Vector.field(1, ti.f32, ng)  # charge density
e = ti.Vector.field(1, ti.f32, ng)  # electric fields
# to show x-vx on the screen
v_x_pos1 = ti.Vector.field(2, ti.f32, int(np / 2))
v_x_pos2 = ti.Vector.field(2, ti.f32, int(np / 2))


@ti.kernel
def initialize():
    for p in x:
        x[p].x = (p + 1) * L / np
        v[p].x = vt * ti.randn() + (-1) ** p * vb  # two streams


@ti.kernel
def substep():
    for p in x:
        x[p] += v[p] * dt
        if x[p].x >= L:  # periodic boundary condition
            x[p] += -L
        if x[p].x < 0:
            x[p] += L
    rho.fill(rho_back)  # fill rho with background charge density
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - 0.5 - base.cast(float)
        rho[base] += (1.0 - fx) * q * inv_dx
        if base[0] < ng - 1:
            rho[base + 1] += fx * q * inv_dx
    e.fill(0.0)
    ti.loop_config(serialize=True)
    for i in range(ng):  # compute electric fields
        if i == 0:
            e[i] = rho[i] * dx * 0.5
        else:
            e[i] = e[i - 1] + (rho[i - 1] + rho[i]) * dx * 0.5
    s = 0.0
    for i in e:
        s += e[i].x
    for i in e:
        e[i] += -s / ng
    for p in v:  # G2P
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - 0.5 - base.cast(float)
        a = e[base] * (1.0 - fx) * qm
        if base[0] < ng - 1:
            a += e[base + 1] * fx * qm  # compute electric force
        v[p] += a * dt


@ti.kernel
def vx_pos():  # to show x-vx on the screen
    for p in x:
        if p % 2:
            v_x_pos1[int((p - 1) / 2)].x = x[p].x / L
            v_x_pos1[int((p - 1) / 2)].y = (v[p].x) / 10 + 0.5
        else:
            v_x_pos2[int(p / 2)].x = x[p].x / L
            v_x_pos2[int(p / 2)].y = (v[p].x) / 10 + 0.5


def main():
    initialize()
    gui = ti.GUI("Shortest PIC", (800, 800))
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(substepping):
            substep()
        vx_pos()
        gui.circles(v_x_pos1.to_numpy(), color=0x0000FF, radius=2)
        gui.circles(v_x_pos2.to_numpy(), color=0xFF0000, radius=2)
        gui.show()


if __name__ == "__main__":
    main()
