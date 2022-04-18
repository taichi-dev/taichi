# Water wave effect partially based on shallow water equations
# https://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

gravity = 6.0  # larger gravity makes wave propagates faster
damping = 0.2  # larger damping makes wave vanishes faster when propagating
dx = 0.02
dt = 0.01
shape = 512, 512
pixels = ti.Vector.field(3, dtype=float, shape=shape)
caustics = ti.field(dtype=float, shape=shape)
height = ti.field(dtype=float, shape=shape)
velocity = ti.field(dtype=float, shape=shape)

surface_height = 10.0

@ti.func
def get_background(p):
    scale = np.pi / (16.0)
    t = ti.sin(p.x * scale) * ti.sin(p.y * scale)
    out = 0.3 + ti.max(0.0, ti.min(1.0, t * 4.0 + 0.5)) * 0.7
    return out * ti.Vector([0.2, 0.6, 0.9])

@ti.kernel
def reset():
    for i, j in height:
        height[i, j] = 0
        velocity[i, j] = 0


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
def get_surface_pos(i, j):
    h = height[i, j] + surface_height
    return ti.Vector([i, h, j])

@ti.func
def get_normal(i, j):
    p0 = get_surface_pos(i, j)
    px = get_surface_pos(i + 1, j)
    py = get_surface_pos(i, j + 1)
    vx = px - p0
    vy = py - p0
    n = vx.cross(vy)
    return n / n.norm()

@ti.func
def refract(I, N, eta):
    k = 1.0 - eta * eta * (1.0 - N.dot(I) * N.dot(I))
    R = ti.Vector([0.0, 0.0, 0.0])
    if k >= 0.0:
        R = eta * I - (eta * N.dot(I) + ti.sqrt(k)) * N
    return R

@ti.func
def isect_ground(p, v):
    p += p.y * v / -v.y
    return p

@ti.kernel
def create_wave(amplitude: ti.f32, x: ti.f32, y: ti.f32):
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        r2 = (i - x)**2 + (j - y)**2
        height[i, j] += amplitude * ti.exp(-0.01 * r2)


@ti.kernel
def update():
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        acceleration = gravity * laplacian(i, j) - damping * velocity[i, j]
        velocity[i, j] = velocity[i, j] + acceleration * dt
        height[i, j] = height[i, j] + velocity[i, j] * dt


@ti.kernel
def visualize_wave():
    # visualizes the wave using refraction and caustics
    for i, j in caustics:
        caustics[i, j] = 0.0
    # Compute normals & refractions, accumulate caustics
    for i, j in pixels:
        p = get_surface_pos(i, j)
        n = get_normal(i, j)
        r = refract(ti.Vector([0, -1, 0]), n, 1.0 / 1.3)
        ground_p = isect_ground(p, r)
        color = get_background(ti.Vector([ground_p.x, ground_p.z]))
        pixels[i, j] = color
        ground_index = ti.cast(ti.Vector([ground_p.x, ground_p.z]), ti.i32)
        caustics[ground_index] += 1.0
    # Filter caustics (Seperable kernel X)
    for i, j in caustics:
        c = 0.0
        w_sum = 0.0
        for subi in ti.ndrange(7):
            offset = ti.Vector([subi, 0]) - 3
            w = ti.exp(-3.0 * ti.sqrt(offset.x * offset.x + offset.y * offset.y))
            c += caustics[ti.Vector([i, j]) + offset] * w
            w_sum += w
        caustics[i, j] = c / w_sum
    # Filter caustics (Seperable kernel Y)
    for i, j in caustics:
        c = 0.0
        w_sum = 0.0
        for subj in ti.ndrange(7):
            offset = ti.Vector([0, subj]) - 3
            w = ti.exp(-3.0 * ti.sqrt(offset.x * offset.x + offset.y * offset.y))
            c += caustics[ti.Vector([i, j]) + offset] * w
            w_sum += w
        caustics[i, j] = c / w_sum
    # Visualize caustics
    for i, j in pixels:
        pixels[i, j] *= 0.3 + 0.7 * caustics[i, j]


print("[Hint] click on the window to create waves")

reset()
window = ti.ui.Window('Water Wave & Caustics', shape, vsync=True)
canvas = window.get_canvas()

while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key == 'r':
            reset()
    if window.is_pressed(ti.ui.LMB):
        key_x, key_y = window.get_cursor_pos()
        create_wave(3.0, key_x * shape[0], key_y * shape[1])
    update()
    visualize_wave()
    canvas.set_image(pixels)
    window.show()
