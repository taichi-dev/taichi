# waterwave.py

from random import randrange, random
import taichi as ti
import numpy as np

ti.init(arch=ti.opengl)

bg = 'misc/test_image.png'
bg = 'misc/test_small.png'
bg = ti.imread(bg, 3)
bg = np.sum(bg, axis=2)
bg = bg.astype(np.float32) / (256 * 3)
bg = np.flip(bg, axis=0)
bg = bg.transpose()

light_color = 1
kappa = 2
gamma = 0.6
eta = 1.333
inv_eta = 1 / eta
inv_eta2 = inv_eta**2
height = 2.5
dx = 0.02
dt = 0.01
inv_dx = 1 / dx
inv_dx2 = inv_dx**2
shape = bg.shape
aspect = shape[0] / shape[1]
n = shape[0]
inv_aspect = 1 / aspect
inv_n = 1 / n
pixels = ti.var(dt=ti.f32, shape=shape)
background = ti.var(dt=ti.f32, shape=shape)
position = ti.var(dt=ti.f32, shape=shape)
velocity = ti.var(dt=ti.f32, shape=shape)
acceleration = ti.var(dt=ti.f32, shape=shape)
background.from_numpy(bg)


@ti.func
def laplacian(i, j):
    return inv_dx2 * (-4 * position[i, j] + position[i, j - 1] +
                      position[i, j + 1] + position[i + 1, j] +
                      position[i - 1, j]) / 4


@ti.func
def gradient(i, j):
    return ti.Vector([
        position[i + 1, j] - position[i - 1, j],
        position[i, j + 1] - position[i, j - 1]
    ]) * inv_dx


@ti.func
def take_linear(i, j):
    m, n = int(i), int(j)
    i, j = i - m, j - n
    return (i * j * background[m + 1, n + 1] +
            (1 - i) * j * background[m, n + 1] + i *
            (1 - j) * background[m + 1, n] + (1 - i) *
            (1 - j) * background[m, n])


@ti.kernel
def touch_at(hurt: ti.f32, x: ti.f32, y: ti.f32):
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        r2 = ti.sqr(i - x) + ti.sqr(j - y)
        position[i, j] = position[i, j] + hurt * ti.exp(-0.02 * r2)


@ti.kernel
def update():
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        acceleration[i, j] = kappa * laplacian(i, j) - gamma * velocity[i, j]

    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        velocity[i, j] = velocity[i, j] + acceleration[i, j] * dt
        position[i, j] = position[i, j] + velocity[i, j] * dt


@ti.kernel
def paint():
    for i, j in pixels:
        g = gradient(i, j)
        # https://www.jianshu.com/p/66a40b06b436
        cos_i = 1 / ti.sqrt(1 + g.norm_sqr())
        cos_o = ti.sqrt(1 - (1 - ti.sqr(cos_i)) * inv_eta2)
        fr = pow(1 - cos_i, 5)
        coh = cos_o * height
        k, l = g[0] * coh, g[1] * coh
        color = take_linear(i + k, j + l)
        pixels[i, j] = (1 - fr) * color + fr * light_color


i = 0
gui = ti.GUI("Water Wave", shape)
while not gui.has_key_pressed():
    i += 1
    if i % 8 == 0 and randrange(8) == 0:
        hurt = random() * 2
        x, y = randrange(8, shape[0] - 8), randrange(8, shape[1] - 8)
        touch_at(hurt, x, y)
    update()
    paint()
    gui.set_image(pixels)
    gui.show()
