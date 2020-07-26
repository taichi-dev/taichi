import taichi as ti
import math
ti.init(ti.cuda)

n = 512
steps = 32
eps = 1e-5

b = ti.field(ti.f32, (n, n))
x = ti.field(ti.f32, (n, n))
d = ti.field(ti.f32, (n, n))
r = ti.field(ti.f32, (n, n))

@ti.func
def c(x: ti.template(), i, j):
    return x[i, j] if 0 <= i < n and 0 <= j < n else 0.0

@ti.func
def A(x: ti.template(), I):
    i, j = I
    return x[i, j] * 4 - c(x, i - 1, j) - c(x, i + 1, j) \
            - c(x, i, j - 1) - c(x, i, j + 1)

@ti.kernel
def init():
    for i in ti.grouped(x):
        d[i] = b[i] - A(x, i)
        r[i] = d[i]

@ti.kernel
def substep():
    alpha, beta, dAd = eps, eps, eps
    for i in ti.grouped(x):
        dAd += d[i] * A(d, i)
    for i in ti.grouped(x):
        alpha += r[i]**2 / dAd
    for i in ti.grouped(x):
        x[i] = x[i] + alpha * d[i]
        r[i] = r[i] - alpha * A(d, i)
        beta += r[i]**2 / (alpha * dAd)
    for i in ti.grouped(x):
        d[i] = r[i] + beta * d[i]

gui = ti.GUI('Possion Solver', (n, n))
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.LMB:
            b[int(e.pos[0] * 512), int(e.pos[1] * 512)] += 0.75
            init()
    for i in range(steps):
        substep()
    gui.set_image(x)
    gui.show()
