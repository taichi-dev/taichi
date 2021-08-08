import math

import taichi as ti

ti.init(arch=ti.gpu)

N = 1600
r0 = 0.05
dt = 1e-5
steps = 160
eps = 1e-3
G = -1e1

pos = ti.Vector.var(2, ti.f32, N)
vel = ti.Vector.var(2, ti.f32, N)


@ti.kernel
def initialize():
    for i in range(N):
        a = ti.random() * math.tau
        r = ti.sqrt(ti.random()) * 0.3
        pos[i] = 0.5 + ti.Vector([ti.cos(a), ti.sin(a)]) * r


@ti.kernel
def substep():
    for i in range(N):
        acc = ti.Vector([0.0, 0.0])

        p = pos[i]
        for j in range(N):
            if i != j:
                r = p - pos[j]
                x = r0 / r.norm(1e-4)
                # Molecular force: https://www.zhihu.com/question/38966526
                acc += eps * (x**13 - x**7) * r
                # Long-distance gravity force:
                acc += G * (x**3) * r

        vel[i] += acc * dt

    for i in range(N):
        pos[i] += vel[i] * dt


gui = ti.GUI('N-body Star')

initialize()
while gui.running and not gui.get_event(ti.GUI.ESCAPE):
    gui.circles(pos.to_numpy(), radius=2, color=0xfbfcbf)
    gui.show()
    for i in range(steps):
        substep()
