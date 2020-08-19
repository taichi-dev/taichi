import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

light_color = 1
kappa = 2
gamma = 0.2
eta = 1.333
depth = 4
dx = 0.02
dt = 0.01
shape = 512, 512
pixels = ti.field(dtype=float, shape=shape)
background = ti.field(dtype=float, shape=shape)
height = ti.field(dtype=float, shape=shape)
velocity = ti.field(dtype=float, shape=shape)
acceleration = ti.field(dtype=float, shape=shape)


@ti.kernel
def reset():
    for i, j in height:
        t = i // 16 + j // 16
        background[i, j] = (t * 0.5) % 1.0
        height[i, j] = 0
        velocity[i, j] = 0
        acceleration[i, j] = 0


@ti.func
def laplacian(i, j):
    return (-4 * height[i, j] + height[i, j - 1] + height[i, j + 1] +
            height[i + 1, j] + height[i - 1, j]) / (4 * dx**2)


@ti.func
def gradient(i, j):
    return ti.Vector([
        height[i + 1, j] - height[i - 1, j],
        height[i, j + 1] - height[i, j - 1]
    ]) * (0.5 / dx)


@ti.func
def take_linear(i, j):
    m, n = int(i), int(j)
    i, j = i - m, j - n
    ret = 0.0
    if 0 <= i < shape[0] and 0 <= i < shape[1]:
        ret = (i * j * background[m + 1, n + 1] +
               (1 - i) * j * background[m, n + 1] + i *
               (1 - j) * background[m + 1, n] + (1 - i) *
               (1 - j) * background[m, n])
    return ret


@ti.kernel
def touch_at(hurt: ti.f32, x: ti.f32, y: ti.f32):
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        r2 = (i - x)**2 + (j - y)**2
        height[i, j] = height[i, j] + hurt * ti.exp(-0.02 * r2)


@ti.kernel
def update():
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        acceleration[i, j] = kappa * laplacian(i, j) - gamma * velocity[i, j]

    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        velocity[i, j] = velocity[i, j] + acceleration[i, j] * dt
        height[i, j] = height[i, j] + velocity[i, j] * dt


@ti.kernel
def paint():
    for i, j in pixels:
        g = gradient(i, j)
        # https://www.jianshu.com/p/66a40b06b436
        cos_i = 1 / ti.sqrt(1 + g.norm_sqr())
        cos_o = ti.sqrt(1 - (1 - (cos_i)**2) * (1 / eta**2))
        fr = pow(1 - cos_i, 2)
        coh = cos_o * depth
        g = g * coh
        k, l = g[0], g[1]
        color = take_linear(i + k, j + l)
        pixels[i, j] = (1 - fr) * color + fr * light_color


print("[Hint] click on the window to create wavelet")

reset()
gui = ti.GUI('Water Wave', shape)
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            gui.running = False
        elif e.key == 'r':
            reset()
        elif e.key == ti.GUI.LMB:
            x, y = e.pos
            touch_at(3, x * shape[0], y * shape[1])
    update()
    paint()
    gui.set_image(pixels)
    gui.show()
